# Copyright (C) 2024 Friedrich Miescher Institute for Biomedical Research

##############################################################################
#                                                                            #
# Author: Nicole Repina              <nicole.repina@fmi.ch>                  #
#                                                                            #
##############################################################################


"""
Expand labels in 2D or 3D image without overlap.
"""
import logging
from typing import Any, Union

import anndata as ad
import dask.array as da
import numpy as np
from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.pyramids import build_pyramid
from fractal_tasks_core.roi import (
    check_valid_ROI_indices,
    convert_indices_to_regions,
    convert_ROI_table_to_indices,
    load_region,
)
from pydantic import validate_call
from zarr.errors import ArrayNotFoundError

from scmultiplex.fractal.fractal_helper_functions import (
    initialize_new_label,
    save_new_label_with_overlap,
)
from scmultiplex.meshing.FilterFunctions import mask_by_parent_object
from scmultiplex.meshing.LabelFusionFunctions import run_expansion

logger = logging.getLogger(__name__)


@validate_call
def expand_labels(
    *,
    # Fractal arguments
    zarr_url: str,
    # Task-specific arguments
    label_name_to_expand: str = "nuc",
    group_by: Union[str, None] = None,
    roi_table: str = "org_ROI_table_linked",
    expand_by_pixels: Union[int, None] = None,
    calculate_image_based_expansion_distance: bool = False,
    expand_by_factor: Union[float, None] = None,
    mask_expansion_by_parent: bool = False,
) -> dict[str, Any]:
    """
    Expand labels in 2D or 3D image without overlap.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            Refers to the zarr_url of the reference acquisition.
            (standard argument for Fractal tasks, managed by Fractal server).
        label_name_to_expand: Label name of segmentation to be expanded.
        group_by: Label name of segmentated objects that are parents of label_name. If None (default), no grouping
            is applied and expansion is calculated for the input object (label_name_to_expand).
            Instead, if a group_by label is specified, the
            label_name_to_expand objects will be masked and grouped by this object. For example, when group_by = 'org',
            the nuclear segmentation is masked by the organoid parent and all nuclei belonging to the parent are loaded
            as a label image.
        roi_table: Name of the ROI table used to iterate over objects and load object regions. If group_by is passed,
            this is the ROI table for the group_by objects, e.g. org_ROI_table.
        expand_by_pixels: Integer value for pixel distance to expand by.
        calculate_image_based_expansion_distance: If true, overrides any set expand_by_pixels value, and expansion
            distance is calculated based on the average label size in loaded region. In this case, expandby_factor must
            be supplied.
        expand_by_factor: Multiplier that specifies pixels by which to expand each label. Float in range
            [0, 1 or higher], e.g. 0.2 means that 20% of mean equivalent diameter of labels in region is used.
        mask_expansion_by_parent: If True, final expanded labels are masked by group_by object. Recommended to set
            to True for child/parent masking.
    """

    logger.info(
        f"Running for {zarr_url=}. \n" f"Label expansion for {label_name_to_expand=}."
    )
    # TODO: for NGIO refactor, this task follows logic of surface_mesh_multiscale task
    # TODO: add integration tests
    # TODO: check that this also runs on MIP full well org seg

    # Check correct task inputs:
    if calculate_image_based_expansion_distance:
        if expand_by_factor is None:
            raise ValueError(
                "Expand-by Factor is missing for image-based calculation of expansion distance. "
                "Check Fractal Task inputs."
            )
        else:
            logger.info(
                f"Running expansion using image-based expansion distance "
                f"estimate with {expand_by_factor=}"
            )
    else:
        if expand_by_pixels is None:
            raise ValueError(
                "Expand by Pixels value is missing. Check Fractal Task inputs."
            )
        else:
            logger.info(
                f"Running expansion using set pixel expansion distance of {expand_by_pixels=}"
            )

    # always use highest resolution label
    level = 0

    # Lazily load zarr array for label image to expand
    # If label does not exist in zarr url, the url is skipped
    try:
        label_dask = da.from_zarr(f"{zarr_url}/labels/{label_name_to_expand}/{level}")
    except ArrayNotFoundError as e:
        logger.warning(
            "Label not found, exit from the task for this zarr url.\n"
            f"Original error: {str(e)}"
        )
        return {}

    # Read ROIs of objects
    roi_adata = ad.read_zarr(f"{zarr_url}/tables/{roi_table}")

    # Read Zarr metadata of label to expand
    label_ngffmeta = load_NgffImageMeta(f"{zarr_url}/labels/{label_name_to_expand}")
    label_xycoars = (
        label_ngffmeta.coarsening_xy
    )  # need to know when building new pyramids
    label_pixmeta = label_ngffmeta.get_pixel_sizes_zyx(level=level)

    # Create list of indices for 3D ROIs spanning the entire Z direction
    # Note that this ROI list is generated based on the input ROI table; if the input ROI table is for the group_by
    # objects, then label regions will be loaded based on the group_by ROIs
    roi_idlist = convert_ROI_table_to_indices(
        roi_adata,
        level=level,
        coarsening_xy=label_xycoars,
        full_res_pxl_sizes_zyx=label_pixmeta,
    )

    check_valid_ROI_indices(roi_idlist, roi_table)

    if len(roi_idlist) == 0:
        logger.warning("Well contains no objects")

    # Initialize parameters to save the newly calculated label map
    # Save with same dimensions as child labels from which they are calculated

    output_label_name = f"{label_name_to_expand}_expanded"
    # output_roi_table_name = f"{label_name_to_expand}_ROI_table_expanded"

    shape = label_dask.shape
    chunks = label_dask.chunksize

    new_label3d_array = initialize_new_label(
        zarr_url,
        shape,
        chunks,
        np.uint32,
        label_name_to_expand,
        output_label_name,
        logger,
    )

    logger.info(
        f"New array saved as {output_label_name=} will have shape {shape} and chunks {chunks}"
    )

    # initialize new ROI table
    # bbox_dataframe_list = []

    ##############
    # Filter nuclei by parent mask ###
    ##############

    if group_by is not None:
        # Load group_by object segmentation to mask child objects by parent group_by object
        # Load well image as dask array for parent objects
        groupby_dask = da.from_zarr(f"{zarr_url}/labels/{group_by}/{level}")

        # Read Zarr metadata
        groupby_ngffmeta = load_NgffImageMeta(f"{zarr_url}/labels/{group_by}")
        groupby_xycoars = groupby_ngffmeta.coarsening_xy
        groupby_pixmeta = groupby_ngffmeta.get_pixel_sizes_zyx(level=level)

        groupby_idlist = convert_ROI_table_to_indices(
            roi_adata,
            level=level,
            coarsening_xy=groupby_xycoars,
            full_res_pxl_sizes_zyx=groupby_pixmeta,
        )

        check_valid_ROI_indices(groupby_idlist, roi_table)

    # Get labels to iterate over
    # TODO handle case when user input well_ROI_table
    roi_labels = roi_adata.obs_vector("label")
    total_label_count = len(roi_labels)
    compute = True

    logger.info(
        f"Starting iteration over {total_label_count} detected objects in ROI table."
    )

    # For each object in input ROI table...
    for row in roi_adata.obs_names:
        row_int = int(row)
        label_str = roi_labels[row_int]
        region = convert_indices_to_regions(roi_idlist[row_int])

        # Load label image of object to expand as numpy array
        seg = load_region(
            data_zyx=label_dask,
            region=region,
            compute=compute,
        )

        if group_by is not None:
            # Mask objects by parent group_by object
            seg, parent_mask = mask_by_parent_object(
                seg, groupby_dask, groupby_idlist, row_int, label_str
            )
        else:
            # Check that label exists in object
            if float(label_str) not in seg:
                raise ValueError(
                    f"Object ID {label_str} does not exist in loaded segmentation image. Does input ROI "
                    f"table match label map?"
                )
            # Select label that corresponds to current object, set all other objects to 0
            seg[seg != float(label_str)] = 0

        ##############
        # Perform label expansion  ###
        ##############

        if calculate_image_based_expansion_distance:
            expandby = expand_by_factor
        else:
            expandby = expand_by_pixels

        seg_expanded, distance = run_expansion(
            seg,
            expandby,
            expansion_distance_image_based=calculate_image_based_expansion_distance,
        )

        if mask_expansion_by_parent and group_by is not None:
            seg_expanded = seg_expanded * parent_mask

        logger.info(f"Expanded label(s) in region {label_str} by {distance} pixels.")

        ##############
        # Save labels and make ROI table ###
        ##############

        # Store labels as new label map in zarr
        # Note that pixels of overlap in the case where two labelmaps are touching are overwritten by the last
        # written object

        save_new_label_with_overlap(
            seg_expanded,
            new_label3d_array,
            zarr_url,
            output_label_name,
            region,
            compute,
        )

    # Starting from on-disk highest-resolution data, build and write to disk a pyramid of coarser levels
    build_pyramid(
        zarrurl=f"{zarr_url}/labels/{output_label_name}",
        overwrite=True,
        num_levels=label_ngffmeta.num_levels,
        coarsening_xy=label_ngffmeta.coarsening_xy,
        chunksize=chunks,
        aggregation_function=np.max,
    )

    logger.info(
        f"Built a pyramid for the {zarr_url}/labels/{output_label_name} label image"
    )

    # TODO: save ROI table

    logger.info(f"End expand_labels task for {zarr_url}/labels/{label_name_to_expand}")

    return {}


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=expand_labels,
        logger_name=logger.name,
    )
