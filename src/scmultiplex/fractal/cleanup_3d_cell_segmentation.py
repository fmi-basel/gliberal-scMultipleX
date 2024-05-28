# Copyright 2024 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Nicole Repina <nicole.repina@fmi.ch>
#

"""
Clean up 3D cell-level segmentation (e.g. nuclei, cells) by size filtering
and disconnected component analysis
"""
from pathlib import Path
from typing import Any
from typing import Sequence

import anndata as ad
import dask.array as da
import logging
import numpy as np
import os
import pandas as pd

import zarr
from fractal_tasks_core.labels import prepare_label_group
from fractal_tasks_core.pyramids import build_pyramid
from fractal_tasks_core.upscale_array import upscale_array
from pydantic.decorator import validate_arguments

from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.roi import (
    check_valid_ROI_indices,
    convert_indices_to_regions,
    convert_ROI_table_to_indices,
    load_region)

from skimage.measure import label, regionprops_table

from scmultiplex.fractal.FractalHelperFunctions import get_zattrs
from scmultiplex.linking.NucleiLinkingFunctions import remove_labels

from scmultiplex.meshing.FilterFunctions import filter_small_sizes_per_round, mask_by_parent_object

logger = logging.getLogger(__name__)


@validate_arguments
def cleanup_3d_cell_segmentation(
        *,
        # Fractal arguments
        input_paths: Sequence[str],
        output_path: str,
        component: str,
        metadata: dict[str, Any],
        # Task-specific arguments
        label_name_toclean: str = "nuc",
        roi_table_toclean: str = "nuc_ROI_table",
        label_name_parent: str = "org_linked",
        roi_table_parent: str = "org_ROI_table_linked",
        level: int = 0,
        volume_filter_threshold: float = 0.05,
        mask_by_parent: bool = True,
        volume_filter: bool = True,
        disconnected_component_filter: bool = True,

) -> dict[str, Any]:
    """
    Calculate 3D surface mesh of parent object (e.g. tissue, organoid)
    from 3D cell-level segmentation of children (e.g. nuclei)

    This task consists of 4 parts:

    1. Load the sub-object (e.g. nuc) segmentation images for each object of a given reference round; skip other rounds.
        Select sub-objects (e.g. nuc) that belong to parent object region by masking by parent.
        Filter the sub-objects to remove small debris that was segmented.
    2. Calculate median value of distribution of cell volumes for a given parent object. Filter volumes less than
        specified cutoff value
    3. disconnected_component_filter

    Parallelization level: image

    Args:
        input_paths: List of input paths where the image data is stored as
            OME-Zarrs. Should point to the parent folder containing one or many
            OME-Zarr files, not the actual OME-Zarr file. Example:
            `["/some/path/"]`. This task only supports a single input path.
            (standard argument for Fractal tasks, managed by Fractal server).
        output_path: This parameter is not used by this task.
            (standard argument for Fractal tasks, managed by Fractal server).
        component: Path to the OME-Zarr image in the OME-Zarr plate that is
            processed. Example: `"some_plate.zarr/B/03/1"`.
            (standard argument for Fractal tasks, managed by Fractal server).
        metadata: This parameter is not used by this task.
            (standard argument for Fractal tasks, managed by Fractal server).
        label_name_toclean: Label name to clean up, e.g. `nuc`.
        roi_table_toclean: Corresponding ROI table name to clean up, e.g"nuc_ROI_table",
        label_name_parent: Label name of segmented objects that is parent of
            label_name e.g. `org_consensus`.
        roi_table_parent: Name of the ROI table over which the task loops to
            calculate the registration. e.g. consensus object table 'org_ROI_table_consensus'
        level: Pyramid level of the labels to register. Choose `0` to
            process at full resolution.
        volume_filter_threshold: discard segmentations (e.g. segmented debris) that have a volume less than
            the volume_filter_threshold * median volume of segmentations composing a given parent object.
            float in range [0,1]. E.g. volume_filter_threshold = 0.05 means that cells with less than 5% of the
            median cell volume (calculated from the distribution of cell volumes composing a given
            parent object) are discarded.
        mask_by_parent: if True, nuclei are masked by parent object (e.g. organoid) to only select nuclei
            belonging to parent. Recommended to set to True when iterating over object (e.g. organoid) ROIs.
        volume_filter: if True, discard segmentations (e.g. segmented debris) below a given volume,
            as specified by volume_filter_threshold
        disconnected_component_filter: if True, discard disconnected components

    """
    logger.info(
        f"Running for {input_paths=}, {component=}. \n"
        f"Cleaning up {roi_table_toclean=} and "
        f"{label_name_toclean=}."
    )
    # TODO: remove level variable from all tasks

    # Set OME-Zarr paths
    input_zarr_path = Path(input_paths[0]) / component

    # Run for all multiplexing rounds that have a matching segmentation and roi table
    current_cycle = input_zarr_path.name

    seg_label_path = f"{input_zarr_path}/labels/{label_name_toclean}/{level}"
    seg_roi_path = f"{input_zarr_path}/tables/{roi_table_toclean}"
    parent_label_path = f"{input_zarr_path}/labels/{label_name_parent}/{level}"
    parent_roi_path = f"{input_zarr_path}/tables/{roi_table_parent}"

    if not os.path.exists(seg_label_path):
        logger.warning(f"Segmentation with label name {label_name_toclean} does not exist in round {current_cycle} "
                       f"of zarr {input_zarr_path} \n"
                       f"Skipping round {current_cycle}")
        return {}

    if os.path.exists(seg_label_path) and not os.path.exists(seg_roi_path):
        logger.warning(f"Roi table with name {roi_table_toclean} does not exist in round {current_cycle} for "
                       f"segmentation with label name {label_name_toclean} of zarr {input_zarr_path} \n"
                       f"Skipping round {current_cycle}")
        return {}

    # Lazily load well image to clean as dask array e.g. for nuclear segmentation, and read ROIs
    logger.info(f"Clean cell segmentation is running for cycle {current_cycle}")
    seg_dask = da.from_zarr(seg_label_path)
    seg_adata = ad.read_zarr(seg_roi_path)
    parent_adata = ad.read_zarr(parent_roi_path)


    # Read Zarr metadata
    seg_ngffmeta = load_NgffImageMeta(f"{input_zarr_path}/labels/{label_name_toclean}")
    seg_xycoars = seg_ngffmeta.coarsening_xy # need to know when building new pyramids
    seg_pixmeta = seg_ngffmeta.get_pixel_sizes_zyx(level=level)

    # Create list of indices for 3D ROIs spanning the entire Z direction
    # TODO is it correct to use nuc label image metadata here?
    parent_idlist_segmeta = convert_ROI_table_to_indices(
        parent_adata,
        level=level,
        coarsening_xy=seg_xycoars,
        full_res_pxl_sizes_zyx=seg_pixmeta,
    )

    check_valid_ROI_indices(parent_idlist_segmeta, roi_table_parent)

    if len(parent_idlist_segmeta) == 0:
        logger.warning("Well contains no objects")

    ##############
    # Initialize new zarr for filtered label image ###
    ##############

    # save as same dimensions as label_name_toclean from which they are calculated
    shape = seg_dask.shape
    chunks = seg_dask.chunksize
    label_dtype = np.uint32
    output_label_name = label_name_toclean + '_clean'
    output_roi_table_name = roi_table_toclean + '_clean'
    store = zarr.storage.FSStore(f"{input_zarr_path}/labels/{output_label_name}/0")

    if len(shape) != 3 or len(chunks) != 3 or shape[0] == 1:
        raise ValueError('Expecting 3D image')

    # Add metadata to labels group
    # Get the label_attrs correctly
    # Note that the new label metadata matches the input label_name_toclean metadata
    label_attrs = get_zattrs(zarr_url=f"{input_zarr_path}/labels/{label_name_toclean}")
    _ = prepare_label_group(
        image_group=zarr.group(input_zarr_path),
        label_name=output_label_name,
        overwrite=True,
        label_attrs=label_attrs,
        logger=logger,
    )

    new_label3d_array = zarr.create(
        shape=shape,
        chunks=chunks,
        dtype=label_dtype,
        store=store,
        overwrite=True,
        dimension_separator="/",
    )

    logger.info(f"Mask will have shape {shape} and chunks {chunks}")

    ##############
    # Load parent mask ###
    ##############

    # load well image as dask array for parent objects
    parent_dask = da.from_zarr(parent_label_path)

    # Read Zarr metadata
    parent_ngffmeta = load_NgffImageMeta(f"{input_zarr_path}/labels/{label_name_parent}")
    parent_xycoars = parent_ngffmeta.coarsening_xy
    parent_pixmeta = parent_ngffmeta.get_pixel_sizes_zyx(level=level)

    parent_idlist_parentmeta = convert_ROI_table_to_indices(
        parent_adata,
        level=level,
        coarsening_xy=parent_xycoars,
        full_res_pxl_sizes_zyx=parent_pixmeta,
    )

    check_valid_ROI_indices(parent_idlist_parentmeta, roi_table_parent)

    # initialize variables
    parent_labels = parent_adata.obs_vector('label')
    compute = True  # convert to numpy array from dask
    segids_toremove_perwell = [] # list of nuclei ids to remove in the well

    ##############
    # Perform cleanup per object ###
    ##############

    # for each parent object (e.g. organoid) in round...
    for row in parent_adata.obs_names:
        row_int = int(row)
        parent_label_id = parent_labels[row_int]
        region = convert_indices_to_regions(parent_idlist_segmeta[row_int])

        # initialize list of label ids to remove for given object
        segids_toremove = []

        # load nuclear label image for given parent object
        seg = load_region(
            data_zyx=seg_dask,
            region=region,
            compute=compute,
        )

        # mask by parent to only analyze nuclei belonging to the given organoid
        seg = mask_by_parent_object(seg, parent_dask, parent_idlist_parentmeta, row_int, parent_label_id)

        # run regionprops to extract centroids and volume of each nuc label
        # note that all cleanup steps is performed on unscaled image which may have strong z-anisotropy
        # TODO: consider upscaling label image prior to alignment, in cases where z-anisotropy is
        #  extreme upscaling could lead to improved performance

        if volume_filter:

            seg_props = regionprops_table(label_image=seg, properties=('label', 'centroid', 'area'))  # zyx

            # output column order must be: ["label", "x_centroid", "y_centroid", "z_centroid", "volume"]
            # units are in pixels
            seg_props = (pd.DataFrame(seg_props,
                                     columns=['label', 'centroid-2', 'centroid-1', 'centroid-0', 'area'])).to_numpy()

            # discard segmentations that have a volume less than fraction of median nuclear volume (segmented debris)
            (seg_props, remove_volumefilter, removed_size_mean, size_mean, volume_cutoff) = (
                filter_small_sizes_per_round(seg_props, column=-1, threshold=volume_filter_threshold))

            logger.info(f"Performing volume filtering of object {parent_label_id} to "
                        f"remove small debris below {volume_cutoff} pix threshold")

            logger.info(f"Filtered out {len(remove_volumefilter)} cells from object {parent_label_id} that have a "
                        f" mean volume of {removed_size_mean} and correspond to labels \n {remove_volumefilter}"
                       )

            remove_volumefilter = list(remove_volumefilter) # list of float64

            segids_toremove.extend(remove_volumefilter)

        if disconnected_component_filter:
            logger.info(f"Performing disconnected component filtering of object {parent_label_id} "
                        f"to remove clusters outside of main object")
            #TODO
            remove_disconnected = []
            segids_toremove.extend(remove_disconnected)

        ##############
        # Filter flagged labels from labelmap and ROI table  ###
        ##############

        segids_toremove_perwell.extend(segids_toremove)
        datatype = seg.dtype
        seg_filtered = remove_labels(seg, segids_toremove, datatype)

        ##############
        # Save labels  ###
        ##############

        # store labels as new label map in zarr

        # load dask from disk, will contain rois of the previously processed objects within for loop
        new_label3d_dask = da.from_zarr(f"{input_zarr_path}/labels/{output_label_name}/0")
        # load region of current object from disk, will include any previously processed neighboring objects
        seg_ondisk = load_region(
            data_zyx=new_label3d_dask,
            region=region,
            compute=compute,
        )

        # check that dimensions of rois match
        if seg_ondisk.shape != seg_filtered.shape:
            raise ValueError('Filtered label image must match image dimensions of bounding box during saving')

        # use fmax so that if one of the elements being compared is a NaN, then the non-nan element is returned
        seg_filtered_tosave = np.fmax(seg_filtered, seg_ondisk)

        # Compute and store 0-th level of new 3d label map to disk
        da.array(seg_filtered_tosave).to_zarr(
            url=new_label3d_array,
            region=region,
            compute=True,
        )

    # Starting from on-disk highest-resolution data, build and write to disk a
    # pyramid of coarser levels

    build_pyramid(
        zarrurl=f"{input_zarr_path}/labels/{output_label_name}",
        overwrite=True,
        num_levels=seg_ngffmeta.num_levels,
        coarsening_xy=seg_ngffmeta.coarsening_xy,
        chunksize=chunks,
        aggregation_function=np.max,
    )

    logger.info(f"Built a pyramid for the {input_zarr_path}/labels/{output_label_name} label image")

    # Filter ROI table
    # print('fullwell', segids_toremove_perwell)
    logger.info(f"End clean cell segmentation task for {input_zarr_path}/labels/{output_label_name}")

    return {}


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=cleanup_3d_cell_segmentation,
        logger_name=logger.name,
    )
