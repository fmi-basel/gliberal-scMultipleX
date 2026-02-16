# Copyright (C) 2025 Friedrich Miescher Institute for Biomedical Research

##############################################################################
#                                                                            #
# Author: Nicole Repina              <nicole.repina@fmi.ch>                  #
#                                                                            #
##############################################################################
import logging
from typing import Optional, Union

import numpy as np
from ngio import open_ome_zarr_container
from ngio.utils import ngio_logger
from pydantic import validate_call

from scmultiplex.fractal.fractal_helper_functions import (
    roi_to_pixel_slices,
    save_new_label_image_with_overlap,
)
from scmultiplex.linking.NucleiLinkingFunctions import run_relabel_dask
from scmultiplex.meshing.LabelFusionFunctions import (
    fill_holes_by_slice_multi_instance,
    get_label_mapping_dask,
    simple_fuse_labels,
)

# Configure logging
ngio_logger.setLevel("ERROR")
logger = logging.getLogger(__name__)


@validate_call
def fuse_touching_labels(
    # Fractal arguments
    zarr_url: str,
    # Task-specific arguments
    label_name_to_fuse: str,
    new_label_name: Optional[str] = None,
    connectivity: Union[int, None] = None,
    level: int = 0,
    fill_holes: bool = False,
) -> None:
    """
    Fuse touching labels in segmentation images, in 2D or 3D. Connected components are identified during labeling
    based on the connectivity argument. For a more detailed explanation of connectivity, see documentation
    of dask_image.ndmeasure.label() and scipy.ndimage.generate_binary_structure() function. When set to None
    (default), squared connectivity = 1 is used.

    Input is segmentation image with 0 value for background. Anything above 0 is assumed to be a labeled object.
    Touching labels are labeled in numerically increasing order starting from 1 to n, where n is the number of
    connected components (objects) identified.

    This task has been tested for fusion of 2D MIP and 3D segmentation and optimized for performance on large 3D arrays
    using dask.

    Output: the fused label image is saved as a new label in zarr, with default name {label_name_to_fuse}_fused. The
    new ROI table for the fused label image is saved as a masking ROI table, with name
    {new_label_name}_ROI_table.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
        label_name_to_fuse: Label name of segmentation to be fused.
        new_label_name: Optionally new name for expanded label.
            If left None, default is {label_name_to_fuse}_fused
        connectivity: Connectivity determines which elements of the output array belong to the structure and
            thus considered neighbors of the central element during fusion. Accepted values
            are ranging from 1 to input.ndim. If None, defaults to a squared connectivity = 1, where only
            connected center faces are considered neighbors (diagonally connected components are considered separated).
            Maximum connectivity is at input.ndim, when
            any face, edge, or vertex is considered a neighbor. Follows default behavior of
            dask_image.ndmeasure.label, with structuring element generated using
            scipy.ndimage.generate_binary_structure(input.ndim, connectivity).
        level: Defaults to full resolution level 0. However, this may lead to out of memory for fusion of large 3D
            arrays. In this case, setting to a higher pyramid level (e.g. 4) will perform fusion using low-resolution
            pyramid. The level-0 array will then be relabelled according to this mapping. Note that low-res fusion
            may yield different results than full-res fusion, since fine spacing between neighboring objects may be
            lost in the low-level representation!
        fill_holes: if True, the new label image (after fusion) has holes filled by iterating
            over z-slices. Useful for filling any gaps between fused labels.
    """

    logger.info(
        f"Running for {zarr_url=} \n" f"for label image='{label_name_to_fuse}'."
    )

    # Load label image
    ome_zarr = open_ome_zarr_container(zarr_url)

    # If level != 0, load low-res image and perform fusion on it
    label_img_to_fuse = ome_zarr.get_label(label_name_to_fuse, path=str(level))
    label_dask_to_fuse = label_img_to_fuse.get_array(mode="dask")

    logger.info(f"Input shape: {label_dask_to_fuse.shape}")
    logger.info(f"Input chunks: {label_dask_to_fuse.chunks}")

    logger.info("Start building dask graphs.")
    logger.info("Fusing labels...")

    fused_dask, label_count, rank = simple_fuse_labels(label_dask_to_fuse, connectivity)

    if connectivity is None:
        # Default dask_image.ndmeasure.label connectivity if None is squared connectivity equal == 1
        connectivity_logged = 1
    else:
        connectivity_logged = connectivity

    # If level != 0, match fused to unfused image and relabel level-0 array to the fused label IDs
    if level != 0:
        # Link unfused (low-res) dask to fused (low-res) dask
        logger.info("Low-resolution fusion: linking unfused to fused zarr...")
        mapping = get_label_mapping_dask(label_dask_to_fuse, fused_dask)
        fused_label_count = len(set(mapping.values()))

        # Check how many ROIs image should have (based on full-res ROI table), if mismatch give warning
        try:
            label_img_roi_table = ome_zarr.get_table(
                f"{label_name_to_fuse}_ROI_table", check_type="generic_roi_table"
            )
        except KeyError:
            logger.warning(
                f"ROI table corresponding to label image '{label_name_to_fuse}' not found."
            )
        else:
            if len(mapping) != len(label_img_roi_table.rois()):
                logger.warning(
                    "Low-resolution level contains fewer ROIs than full-resolution label image. "
                    "Some small objects may be lost. Consider changing to a higher-resolution level."
                )

        logger.info(f"Computed mapping dictionary: {mapping}")

        # Relabel level-0 image based on mapping
        # Load level-0
        label_img_full_res = ome_zarr.get_label(label_name_to_fuse)
        label_dask_full_res = label_img_full_res.get_array(mode="dask")
        # Relabel
        logger.info("Low-resolution fusion: relabeling level-0 zarr...")
        dask_to_save = run_relabel_dask(label_dask_full_res, matching_dict=mapping)

    else:
        logger.info("Performed full-resolution fusion on level-0 zarr.")
        # if already performed fusion on full-res dask, directly save it
        dask_to_save = fused_dask
        # TODO: get label counts from final ROI table, this will be faster
        # fused_label_count = label_count.compute()
        fused_label_count = "NA"

    logger.info(f"Fused dask array shape: {dask_to_save.shape}")
    logger.info(f"Fused dask chunks: {dask_to_save.chunks}")

    logger.info("Finished building dask graphs.")

    # Save new fused label image and masking ROI table
    if new_label_name is None:
        output_label_name = f"{label_name_to_fuse}_fused"
    else:
        output_label_name = new_label_name

    new_label_container = ome_zarr.derive_label(name=output_label_name, overwrite=True)

    logger.info(f"Target shape: {new_label_container.shape}")
    logger.info(f"Target chunks: {new_label_container.chunks}")

    logger.info("Computing and saving zarr...")
    new_label_container.set_array(dask_to_save)

    logger.info(
        f"Finished compute, found {fused_label_count} connected components using structuring element rank {rank} and "
        f"squared connectivity {connectivity_logged}."
    )

    # Build pyramids for label image
    new_label_container.consolidate()
    logger.info(f"Built a pyramid for the {output_label_name} label image")

    # Make ROI table from new label image
    masking_table = ome_zarr.build_masking_roi_table(output_label_name)
    new_table_name = f"{output_label_name}_ROI_table"
    ome_zarr.add_table(new_table_name, masking_table, overwrite=True)
    logger.info(f"Saved new masking ROI table as {new_table_name}")

    if fill_holes:
        # Loop over ROIs, fill holes, write back to same zarr.
        # Load label image
        ome_zarr = open_ome_zarr_container(
            zarr_url
        )  # re-open container to get fused image
        fused_label_img = ome_zarr.get_masked_label(
            label_name=output_label_name, masking_label_name=output_label_name
        )

        # Load ROI tables
        roi_table = ome_zarr.get_table(new_table_name, check_type="generic_roi_table")

        # Pixel sizes as list: [z,y,x]
        pixel_size = fused_label_img.pixel_size
        spacing = np.array([pixel_size.z, pixel_size.y, pixel_size.x])

        logger.info("Fill holes: Iterating over fused ROIs")

        # Fill holes per ROI
        for roi in roi_table.rois():
            label_string = roi.name
            label_int = int(label_string)
            logger.info(f"Fill holes: Processing ROI label {label_string}.")
            roi_np = fused_label_img.get_roi(
                label=label_int
            )  # load label region as numpy array

            # Fill holes
            roi_np_filled = fill_holes_by_slice_multi_instance(roi_np)

            # Save ROI to disk using dask _to_zarr, not ngio
            region = roi_to_pixel_slices(roi, spacing)
            save_new_label_image_with_overlap(
                roi_np_filled, zarr_url, output_label_name, region
            )
            logger.info(
                f"Fill holes: Wrote region {label_string} to level-0 zarr image."
            )

        # Build pyramids for filled label image
        filled_label_img = ome_zarr.get_label(name=output_label_name)
        filled_label_img.consolidate()
        logger.info(
            f"Fill holes: Built a pyramid for the {output_label_name} label image"
        )

        # Make ROI table from filled label image
        masking_table = ome_zarr.build_masking_roi_table(output_label_name)
        ome_zarr.add_table(new_table_name, masking_table, overwrite=True)
        logger.info(f"Fill holes: Saved new masking ROI table as {new_table_name}")

    logger.info(f"End fuse_touching_labels task for {zarr_url}")

    return


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=fuse_touching_labels,
        logger_name=logger.name,
    )
