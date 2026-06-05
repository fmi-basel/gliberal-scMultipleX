# Copyright 2023 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Nicole Repina <nicole.repina@fmi.ch>
# Tommaso Comparin <tommaso.comparin@exact-lab.it>
# Joel Lüthi <joel.luethi@uzh.ch>
#

"""
Calculates label-based linking between pairs of multiplexing rounds for segmented objects (e.g. nuclei, cells)
that are children of linked parent objects (e.g. organoids)
"""
import logging
import os
import sys
from traceback import print_exc
from typing import Any, Optional

import anndata as ad
import dask.array as da
import numpy as np
import pandas as pd
import SimpleITK as sitk
import zarr
from fractal_tasks_core.channels import (
    ChannelInputModel,
    ChannelNotFoundError,
    OmeroChannel,
    get_channel_from_image_zarr,
)
from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.roi import (
    check_valid_ROI_indices,
    convert_indices_to_regions,
    convert_ROI_table_to_indices,
    load_region,
)
from fractal_tasks_core.tables import write_table
from fractal_tasks_core.tasks.io_models import InitArgsRegistration
from fractal_tasks_core.upscale_array import upscale_array
from pydantic import validate_call
from skimage.measure import regionprops_table

# this is an hack to run platymatch without modifying its code. In some parts of the code
# platymatch will import other submodules from itself in an absolute way (i.e. from platymatch.xxx import ....)
# rather than a relative way. Maybe we can create a pull request to switch those import statements to relative
import scmultiplex
from scmultiplex.fractal.fractal_helper_functions import extract_acq_info
from scmultiplex.linking.NucleiLinkingFunctions import (
    remove_labels,
    run_affine,
    run_ffd,
)
from scmultiplex.meshing.FilterFunctions import filter_small_sizes_per_round

sys.path.append(os.path.join(scmultiplex.__path__[0], r"platymatch"))
from platymatch.utils.utils import generate_affine_transformed_image  # noqa

logger = logging.getLogger(__name__)


@validate_call
def calculate_platymatch_registration(
    *,
    # Fractal arguments
    zarr_url: str,
    init_args: InitArgsRegistration,
    # Task-specific arguments
    seg_channel: ChannelInputModel,
    label_name_to_register: str = "nuc",
    label_name_obj: str = "org_linked",
    roi_table: str = "org_linked_ROI_table",
    new_linking_table_name: Optional[str] = None,
    level: int = 0,
    save_transformation: bool = True,
    mask_by_parent: bool = True,
    calculate_ffd: bool = True,
    volume_filter: bool = True,
    volume_filter_threshold: float = 0.05,
) -> dict[str, Any]:
    """
    Calculate point-cloud-based registration with PlatyMatch.

    This task consists of 4 parts:

    1. Load the sub-object segmentation images for each well (paired reference and alignment round)
    2. Select sub-objects that belong to object region by loading with object ROI table and mask by object mask.
       Object pair is defined by consensus linking. Filter the sub-objects to remove small debris that was segmented.
    3. Calculate affine and optionally the free-form deformation for each object pair
    4. Output: save the identified matches as a linking table in alignment round directory
       and optionally the transformation matrix on disk.

    Parallelization level: image

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        init_args: Intialization arguments provided by
            `_image_based_registration_hcs_init`. They contain the
            reference_zarr_url that is used for registration.
            (standard argument for Fractal tasks, managed by Fractal server).
        label_name_to_register: Label name that will be used for label-based
            registration, e.g. `nuc`.
        label_name_obj: Label name of segmented objects that is parent of
            label_name_to_register e.g. `org_linked`.
        roi_table: Name of the ROI table over which the task loops to
            calculate the registration. e.g. linked consensus object table 'org_ROI_table_linked'
        new_linking_table_name: Optional new table name for linked matches, saved in RX
            alignment round directory. Suffix "_affine" or "_ffd" is added for the respective registration.
            If left None, default is {label_name_to_register} + "_match_table_affine" or "_match_table_ffd"
        level: Pyramid level of the labels to register. Choose `0` to
            process at full resolution.
        save_transformation: if True, saves the transformation matrix on disk in subfolder 'transformations'
        mask_by_parent: if True, nuclei are masked by parent object (e.g. organoid) to only select nuclei
            belonging to parent. Recommended to set to True when iterating over object (e.g. organoid) ROIs.
        calculate_ffd: if True, calculate free form deformation registration based on affine linking.
        seg_channel: Channel that was used for nuclear segmentation; requires either
            `wavelength_id` (e.g. `A01_C01`) or `label` (e.g. `DAPI`). Assume same across all rounds.
        volume_filter: if True, performing volume filtering of nuclei to remove objects smaller
            than specified volume_filter_threshold.
        volume_filter_threshold: Multiplier that specifies cutoff for volumes below which nuclei are filtered out,
            float in range [0,1], e.g. 0.05 means that 5% of median of nuclear volume distribution is used as cutoff.
            Specify this value if volume filtering is desired. Default 0.05.
    """

    r0_zarr_path = init_args.reference_zarr_url
    zarr_acquisition = extract_acq_info(zarr_url)
    ref_acquisition = extract_acq_info(r0_zarr_path)

    logger.info(
        f"Running for {zarr_url=} \n"
        f"relative to reference round {r0_zarr_path} \n"
        f"Calculating translation registration per {roi_table=} for "
        f"{label_name_to_register=}."
    )

    # Lazily load zarr array
    # Reference (e.g. R0, fixed) vs. alignment (e.g. RX, moving)
    # load well image as dask array e.g. for nuclear segmentation
    r0_dask = da.from_zarr(f"{r0_zarr_path}/labels/{label_name_to_register}/{level}")
    rx_dask = da.from_zarr(f"{zarr_url}/labels/{label_name_to_register}/{level}")

    # Read ROIs
    r0_adata = ad.read_zarr(f"{r0_zarr_path}/tables/{roi_table}")
    rx_adata = ad.read_zarr(f"{zarr_url}/tables/{roi_table}")

    # Read Zarr metadata
    r0_ngffmeta = load_NgffImageMeta(f"{r0_zarr_path}/labels/{label_name_to_register}")
    rx_ngffmeta = load_NgffImageMeta(f"{zarr_url}/labels/{label_name_to_register}")
    r0_xycoars = r0_ngffmeta.coarsening_xy  # need to know when building new pyramids
    rx_xycoars = rx_ngffmeta.coarsening_xy
    r0_pixmeta = r0_ngffmeta.get_pixel_sizes_zyx(level=level)
    rx_pixmeta = rx_ngffmeta.get_pixel_sizes_zyx(level=level)

    if r0_pixmeta != rx_pixmeta:
        raise ValueError("Pixel sizes need to be equal between cycles for registration")

    if len(r0_adata) != len(rx_adata):
        raise ValueError(
            "Number of objects does not match between reference object {input_paths=} and alignment round"
        )

    logger.info(
        f"Found {len(r0_adata)} objects in reference and "
        f"{len(rx_adata)} objects in alignment round for registration from table {roi_table=}."
    )

    # Create list of indices for 3D ROIs spanning the entire Z direction
    r0_idlist = convert_ROI_table_to_indices(
        r0_adata,
        level=level,
        coarsening_xy=r0_xycoars,
        full_res_pxl_sizes_zyx=r0_pixmeta,
    )

    rx_idlist = convert_ROI_table_to_indices(
        rx_adata,
        level=level,
        coarsening_xy=rx_xycoars,
        full_res_pxl_sizes_zyx=rx_pixmeta,
    )

    check_valid_ROI_indices(r0_idlist, roi_table)
    check_valid_ROI_indices(rx_idlist, roi_table)

    if len(r0_idlist) == 0 or len(rx_idlist) == 0:
        logger.warning("Well contains no objects")

    if mask_by_parent:
        # load well image as dask array for parent objects
        r0_dask_parent = da.from_zarr(f"{r0_zarr_path}/labels/{label_name_obj}/{level}")
        rx_dask_parent = da.from_zarr(f"{zarr_url}/labels/{label_name_obj}/{level}")

        # Read Zarr metadata
        r0_ngffmeta_parent = load_NgffImageMeta(
            f"{r0_zarr_path}/labels/{label_name_obj}"
        )
        rx_ngffmeta_parent = load_NgffImageMeta(f"{zarr_url}/labels/{label_name_obj}")
        r0_xycoars_parent = r0_ngffmeta_parent.coarsening_xy
        rx_xycoars_parent = rx_ngffmeta_parent.coarsening_xy
        r0_pixmeta_parent = r0_ngffmeta_parent.get_pixel_sizes_zyx(level=level)
        rx_pixmeta_parent = rx_ngffmeta_parent.get_pixel_sizes_zyx(level=level)

        r0_idlist_parent = convert_ROI_table_to_indices(
            r0_adata,
            level=level,
            coarsening_xy=r0_xycoars_parent,
            full_res_pxl_sizes_zyx=r0_pixmeta_parent,
        )

        rx_idlist_parent = convert_ROI_table_to_indices(
            rx_adata,
            level=level,
            coarsening_xy=rx_xycoars_parent,
            full_res_pxl_sizes_zyx=rx_pixmeta_parent,
        )

        check_valid_ROI_indices(r0_idlist_parent, roi_table)
        check_valid_ROI_indices(rx_idlist_parent, roi_table)

    # load raw images for free-form deformation
    if calculate_ffd:
        # load intensity well image as dask array for the segmentation channel
        # TODO: reference and alignment channel index must be the same, add check

        # Find channel index for reference round (r0)
        try:
            tmp_channel: OmeroChannel = get_channel_from_image_zarr(
                image_zarr_path=f"{r0_zarr_path}",
                wavelength_id=seg_channel.wavelength_id,
                label=seg_channel.label,
            )
        except ChannelNotFoundError as e:
            logger.warning(
                "Channel not found, exit from the task.\n" f"Original error: {str(e)}"
            )
            return {}
        r0_channel = tmp_channel.index

        # Find channel index for alignment round (rx)
        try:
            tmp_channel: OmeroChannel = get_channel_from_image_zarr(
                image_zarr_path=f"{zarr_url}",
                wavelength_id=seg_channel.wavelength_id,
                label=seg_channel.label,
            )
        except ChannelNotFoundError as e:
            logger.warning(
                "Channel not found, exit from the task.\n" f"Original error: {str(e)}"
            )
            return {}
        rx_channel = tmp_channel.index

        # Load channel data
        r0_dask_raw = da.from_zarr(f"{r0_zarr_path}/{level}")[r0_channel]
        rx_dask_raw = da.from_zarr(f"{zarr_url}/{level}")[rx_channel]

        # Read Zarr metadata
        r0_ngffmeta_raw = load_NgffImageMeta(f"{r0_zarr_path}")
        rx_ngffmeta_raw = load_NgffImageMeta(f"{zarr_url}")
        r0_xycoars_raw = r0_ngffmeta_raw.coarsening_xy
        rx_xycoars_raw = rx_ngffmeta_raw.coarsening_xy
        r0_pixmeta_raw = r0_ngffmeta_raw.get_pixel_sizes_zyx(level=level)
        rx_pixmeta_raw = rx_ngffmeta_raw.get_pixel_sizes_zyx(level=level)

        r0_idlist_raw = convert_ROI_table_to_indices(
            r0_adata,
            level=level,
            coarsening_xy=r0_xycoars_raw,
            full_res_pxl_sizes_zyx=r0_pixmeta_raw,
        )

        rx_idlist_raw = convert_ROI_table_to_indices(
            rx_adata,
            level=level,
            coarsening_xy=rx_xycoars_raw,
            full_res_pxl_sizes_zyx=rx_pixmeta_raw,
        )

        check_valid_ROI_indices(r0_idlist_raw, roi_table)
        check_valid_ROI_indices(rx_idlist_raw, roi_table)

    ##############
    #  Calculate the transformation
    ##############

    # TODO add check that adata is numerically increasing incrementally
    r0_labels = r0_adata.obs_vector("label")  # unsorted, match row order
    rx_labels = rx_adata.obs_vector("label")  # unsorted, match row order

    # sort labels numerically; if labels were remapped during organoid registration, may not be in order
    r0_labels_sorted = r0_labels[r0_labels.astype(float).argsort()]
    rx_labels_sorted = rx_labels[rx_labels.astype(float).argsort()]

    # initialize variables
    compute = True  # convert to numpy array form dask

    # list of linking dfs for each object, append to this list as generated in for loop
    linked_df_list_affine = []
    linked_df_list_ffd = []

    # TODO make possible to run on FOV roi table as well
    # for each object in r0...
    for r0_org_label, rx_org_label in zip(r0_labels_sorted, rx_labels_sorted):

        transform_affine = None  # clear transform from previous organoid pair

        if r0_org_label != rx_org_label:
            raise ValueError(
                f"Label mismatch between reference object {r0_org_label} of round {ref_acquisition}"
                f"and alignment object {rx_org_label} of round {zarr_acquisition}. \n"
                f"Platymatch registration must be run on consensus-linked objects"
            )

        if (
            len(np.where(r0_labels == r0_org_label)[0]) > 1
            or len(np.where(rx_labels == rx_org_label)[0]) > 1
        ):
            raise ValueError("Organoid label values are not unique")

        logger.info(
            f"Loading images for reference object label {r0_org_label} "
            f"and alignment object label {rx_org_label}"
        )

        # identify index that corresponds to label id
        r0_rowid = np.where(r0_labels == r0_org_label)[0][0]
        rx_rowid = np.where(rx_labels == rx_org_label)[0][0]

        # load nuclear label image for object in r0
        r0 = load_region(
            data_zyx=r0_dask,
            region=convert_indices_to_regions(r0_idlist[r0_rowid]),
            compute=compute,
        )

        # load nuclear label image for matched object in rx
        rx = load_region(
            data_zyx=rx_dask,
            region=convert_indices_to_regions(rx_idlist[rx_rowid]),
            compute=compute,
        )

        if mask_by_parent:
            # load object label image for object in r0
            r0_parent = load_region(
                data_zyx=r0_dask_parent,
                region=convert_indices_to_regions(r0_idlist_parent[r0_rowid]),
                compute=compute,
            )

            # load object label image for object in rx
            rx_parent = load_region(
                data_zyx=rx_dask_parent,
                region=convert_indices_to_regions(rx_idlist_parent[rx_rowid]),
                compute=compute,
            )

            # if object segmentation was run at a different level than nuclear segmentation,
            # need to upscale arrays to match shape
            if r0_parent.shape != r0.shape:
                r0_parent = upscale_array(
                    array=r0_parent, target_shape=r0.shape, pad_with_zeros=False
                )

            if rx_parent.shape != rx.shape:
                rx_parent = upscale_array(
                    array=rx_parent, target_shape=rx.shape, pad_with_zeros=False
                )

            # mask nuclei by parent object
            r0_parent_mask = np.zeros_like(r0_parent)
            rx_parent_mask = np.zeros_like(rx_parent)

            r0_parent_mask[
                r0_parent == int(r0_org_label)
            ] = 1  # select only current object and binarize object mask
            rx_parent_mask[
                rx_parent == int(rx_org_label)
            ] = 1  # select only current object and binarize object mask

            r0 = r0 * r0_parent_mask
            rx = rx * rx_parent_mask

        # run regionprops to extract centroids and volume of each nuc label
        # note that registration is performed on unscaled image
        # TODO: consider upscaling label image prior to alignment, in cases where z-anisotropy is
        #  extreme upscaling could lead to improved performance
        r0_props = regionprops_table(
            label_image=r0, properties=("label", "centroid", "area")
        )  # zyx
        rx_props = regionprops_table(
            label_image=rx, properties=("label", "centroid", "area")
        )

        # output column order must be: ["label", "x_centroid", "y_centroid", "z_centroid", "volume"]
        r0_props = (
            pd.DataFrame(
                r0_props,
                columns=["label", "centroid-2", "centroid-1", "centroid-0", "area"],
            )
        ).to_numpy()
        rx_props = (
            pd.DataFrame(
                rx_props,
                columns=["label", "centroid-2", "centroid-1", "centroid-0", "area"],
            )
        ).to_numpy()

        if volume_filter:
            # discard segmentations that have a volume less than fraction of median nuclear volume (segmented debris)
            # reference round
            (
                r0_props,
                fixed_removed,
                r0_removed_size_mean,
                r0_size_mean,
                r0_volume_cutoff,
            ) = filter_small_sizes_per_round(
                r0_props, column=-1, threshold=volume_filter_threshold
            )

            logger.info(
                f"Volume filtering removed {len(fixed_removed)} cell(s) from object {r0_org_label} of round "
                f"{ref_acquisition} that have a volume below the calculated {r0_volume_cutoff} pixel threshold"
                f"\n Removed labels have a mean volume of {r0_removed_size_mean} and are the label id(s): "
                f"\n {fixed_removed}"
            )
            # rx
            (
                rx_props,
                moving_removed,
                rx_removed_size_mean,
                rx_size_mean,
                rx_volume_cutoff,
            ) = filter_small_sizes_per_round(
                rx_props, column=-1, threshold=volume_filter_threshold
            )

            logger.info(
                f"Volume filtering removed {len(moving_removed)} cell(s) from object {rx_org_label} of round "
                f"{zarr_acquisition} that have a volume below the calculated {rx_volume_cutoff} pixel threshold"
                f"\n Removed labels have a mean volume of {rx_removed_size_mean} and are the label id(s): "
                f"\n {moving_removed}"
            )

        # TODO add disconnected component detection here to remove nuclei that don't belong to main organoid

        ##############
        # Calculate affine linking with Platymatch ###
        ##############

        # if insufficient nuclear count, skip this organoid pair
        if r0_props.shape[0] <= 3 or rx_props.shape[0] <= 3:
            logger.info(
                f"Skipping organoid pair [ reference object {r0_org_label} of round {ref_acquisition}, "
                f"alignment object {rx_org_label} of round {zarr_acquisition} ] of shape "
                f"[{r0_props.shape[0]}, {rx_props.shape[0]}] due to insufficient child object count. "
            )
            continue

        try:
            logger.info(
                f"Trying affine matching for reference object label {r0_org_label} "
                f"and alignment object label {rx_org_label} of round {zarr_acquisition}"
            )

            (affine_matches, transform_affine) = run_affine(
                rx_props, r0_props, ransac_iterations=4000, icp_iterations=50
            )

            logger.info(
                f"Successful affine matching of reference object label {r0_org_label} "
                f"and alignment object label {rx_org_label}"
            )

            affine_matches = pd.DataFrame(
                affine_matches,
                columns=[
                    "R" + str(ref_acquisition) + "_label",
                    "R" + str(zarr_acquisition) + "_label",
                    "pixdist",
                    "confidence",
                ],
            )

            linked_df_list_affine.append(affine_matches)

            if save_transformation and transform_affine is not None:
                # store the transformation matrix on disk
                # check if transformation folder exists, if not create it
                save_transform_path = (
                    f"{zarr_url}/transforms/{roi_table}_{label_name_to_register}_affine"
                )
                os.makedirs(save_transform_path, exist_ok=True)
                # saving name is row in obs_names of tables anndata; so matches with input tables ROI naming
                save_name = f"{r0_org_label}.npy"
                np.save(
                    f"{save_transform_path}/{save_name}",
                    transform_affine,
                    allow_pickle=False,
                )

        except Exception as e:
            print("Exception!", r0_org_label, rx_org_label, e)
            print_exc()

        ##############
        # Calculate free form deformation with Platymatch ###
        ##############

        if calculate_ffd and transform_affine is not None:
            # relabel segmentation images to match object filtering
            # TODO update here if filtering changes (e.g. if add disconnected component removal)

            datatype = r0.dtype
            r0 = remove_labels(r0, fixed_removed, datatype)
            rx = remove_labels(rx, moving_removed, datatype)

            # load nuclear raw image for object in r0
            r0_channel_raw = load_region(
                data_zyx=r0_dask_raw,
                region=convert_indices_to_regions(r0_idlist_raw[r0_rowid]),
                compute=compute,
            )

            # load nuclear raw image for object in rx
            rx_channel_raw = load_region(
                data_zyx=rx_dask_raw,
                region=convert_indices_to_regions(rx_idlist_raw[rx_rowid]),
                compute=compute,
            )

            # TODO: handle case if there is a shape mismatch; must downsample or upsample raw image
            #  to match shape of segmentation image.
            #  This may happen if nuclear segmentation was run at a lower resolution level than input as "level" ...
            #  parameter in calculate_platymatch_registration

            if r0_channel_raw.shape != r0.shape or rx_channel_raw.shape != rx.shape:
                raise ValueError(
                    "Image shape must match between raw and segmentation image"
                )

            try:
                # generate transformed affine images
                (
                    moving_transformed_affine_raw_image,
                    moving_transformed_affine_label_image,
                ) = generate_affine_transformed_image(
                    transform_matrix=transform_affine,
                    fixed_raw_image=r0_channel_raw,
                    moving_raw_image=rx_channel_raw,
                    moving_label_image=rx,
                )

                # run ffd matching
                # for now only use ffd_matches result, do not save transform or transformed image
                (ffd_matches, transform_ffd, transformed_ffd_label_image) = run_ffd(
                    rx_props,
                    r0_props,
                    moving_transformed_affine_raw_image,
                    moving_transformed_affine_label_image,
                    r0_channel_raw,
                    r0,
                )

                logger.info(
                    f"Successful ffd matching of reference object label {r0_org_label} "
                    f"and alignment object label {rx_org_label}"
                )

                ffd_matches = pd.DataFrame(
                    ffd_matches,
                    columns=[
                        "R" + str(ref_acquisition) + "_label",
                        "R" + str(zarr_acquisition) + "_label",
                        "pixdist",
                    ],
                )
                linked_df_list_ffd.append(ffd_matches)

                if save_transformation and transform_ffd is not None:
                    # store the transformation matrix on disk
                    # check if transformation folder exists, if not create it
                    save_transform_path = f"{zarr_url}/transforms/{roi_table}_{label_name_to_register}_ffd"
                    os.makedirs(save_transform_path, exist_ok=True)
                    # saving name is row in obs_names of tables anndata; so matches with input tables ROI naming
                    save_name = f"{r0_org_label}.tfm"
                    sitk.WriteTransform(
                        transform_ffd, f"{save_transform_path}/{save_name}"
                    )

            except Exception as e:
                print("Exception!", r0_org_label, rx_org_label, e)
                print_exc()

    # concatenate list to generate single df for all nuclei in well
    if len(linked_df_list_affine) > 0:
        link_df_affine = pd.concat(linked_df_list_affine, ignore_index=True, sort=False)
    else:
        link_df_affine = None

    if calculate_ffd and len(linked_df_list_ffd) > 0:
        link_df_ffd = pd.concat(linked_df_list_ffd, ignore_index=True, sort=False)
    else:
        link_df_ffd = None

    ##############
    # Storing the calculated affine linking as anndata ###
    ##############

    # TODO refactor into a saving function
    if link_df_affine is not None:
        link_df_adata = ad.AnnData(X=np.array(link_df_affine, dtype=np.float32))
        obsnames = list(map(str, link_df_affine.index))
        varnames = list(link_df_affine.columns.values)
        link_df_adata.obs_names = obsnames
        link_df_adata.var_names = varnames

        if new_linking_table_name is None:
            new_link_table = label_name_to_register + "_match_table_affine"
        else:
            new_link_table = new_linking_table_name + "_affine"

        # Save the linking table as a new table
        logger.info(
            f"Writing the affine linking table for round {zarr_acquisition} as {new_link_table}"
        )

        image_group = zarr.group(f"{zarr_url}")

        write_table(
            image_group,
            new_link_table,
            link_df_adata,
            overwrite=True,
            table_attrs=dict(type="linking_table", fractal_table_version="1"),
        )

    # TODO refactor into a saving function
    if calculate_ffd and link_df_ffd is not None:
        link_df_adata = ad.AnnData(X=np.array(link_df_ffd, dtype=np.float32))
        obsnames = list(map(str, link_df_ffd.index))
        varnames = list(link_df_ffd.columns.values)
        link_df_adata.obs_names = obsnames
        link_df_adata.var_names = varnames

        if new_linking_table_name is None:
            new_link_table = label_name_to_register + "_match_table_ffd"
        else:
            new_link_table = new_linking_table_name + "_ffd"

        # Save the linking table as a new table
        logger.info(
            f"Writing the ffd linking table round {zarr_acquisition} as {new_link_table}"
        )

        image_group = zarr.group(f"{zarr_url}")

        write_table(
            image_group,
            new_link_table,
            link_df_adata,
            overwrite=True,
            table_attrs=dict(type="linking_table", fractal_table_version="1"),
        )

    return {}


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    # from multiprocessing import freeze_support
    #
    # freeze_support()

    run_fractal_task(
        task_function=calculate_platymatch_registration,
        logger_name=logger.name,
    )
