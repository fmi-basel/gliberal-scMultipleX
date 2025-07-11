# Copyright (C) 2024 Friedrich Miescher Institute for Biomedical Research

##############################################################################
#                                                                            #
# Author: Nicole Repina              <nicole.repina@fmi.ch>                  #
#                                                                            #
##############################################################################
import logging
from typing import Optional, Union

import numpy as np
import pandas as pd
from fractal_tasks_core.pyramids import build_pyramid
from fractal_tasks_core.roi import (
    array_to_bounding_box_table,
    convert_indices_to_regions,
    load_region,
)
from pydantic import validate_call

from scmultiplex.fractal.fractal_helper_functions import (
    get_zattrs,
    initialize_new_label,
    load_label_rois,
    save_masking_roi_table_from_df_list,
    save_new_label_with_overlap,
)
from scmultiplex.meshing.FilterFunctions import mask_by_parent_object
from scmultiplex.meshing.LabelFusionFunctions import (
    fill_holes_by_slice_multi_instance,
    run_expansion,
)

logger = logging.getLogger(__name__)


@validate_call
def expand_labels(
    # Fractal arguments
    zarr_url: str,
    # Task-specific arguments
    label_name_to_expand: str = "nuc",
    new_label_name: Optional[str] = None,
    roi_table: str = "org_ROI_table_linked",
    masking_label_map: Union[str, None] = None,
    mask_output: bool = True,
    expand_by_pixels: Union[int, None] = None,
    calculate_image_based_expansion_distance: bool = False,
    expand_by_factor: Union[float, None] = None,
    fill_holes: bool = False,
) -> None:
    """
    Expand labels in 2D or 3D segmentation images in XY. For 3D images, expansion is performed on each 2D
    z-slice iteratively. Thus, labels are only expanded in XY (i.e. laterally, not in z). Labels are grown outwards
    by up to the distance specified by expand_by_pixels or expand_by_factor, without overflowing into
    neighboring regions. See skimage.segmentation.expand_labels() for further documentation.

    Expansion is run on input label_name_to_expand, iterating over regions of input roi_table. It is possible to run
    expansion on the full well image (e.g by specifying well_ROI_table) as input roi_table, or on individual objects
    within image (e.g. by specifying a segmentation masking ROI table) as input roi_table. In the later case, a common
    use case would be to expand in 3D nuclei of each organoid in dataset.

    Output: the expanded label image is saved as a new label in zarr, with name {label_name_to_expand}_expanded. The
    new ROI table for the expanded label image is saved as a masking ROI table, with name
    {label_name_to_expand}_expanded_ROI_table.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
        label_name_to_expand: Label name of segmentation to be expanded.
        new_label_name: Optionally new name for expanded label.
            If left None, default is {label_name_to_expand}_expanded
        roi_table: Name of the ROI table used to iterate over objects and load object regions. If a table of type
            "roi_table" is passed, e.g. well_ROI_table, all objects for each region in the table will be loaded
            and expanded simultaneously. If a table of type "masking_roi_table" is passed, e.g. a segmentation
            ROI table, the task iterates over these objects and loads only the children (i.e. label_name_to_expand) that
            belong to the parent object.
        masking_label_map: Label name of segmented objects that are parents of label_name. This input is
            mandatory if a roi table of type "masking_roi_table" is provided. It is the name of the label map
            that corresponds to the input ROI table. The masking_label_map will be used to mask label_name_to_expand
            objects, to only select children belonging to given parent.
        mask_output: If True, expanded label is masked by parent label. Only used if masking_label_map is provided.
            Recommended to set as True, to avoid overwriting of children labels between neighboring parents. However,
            it may lead to expanded results to be cropped by parent mask; in this case, the parent mask can first be
            expanded.
        expand_by_pixels: Default expansion parameter. Integer value for pixel distance to expand by.
        calculate_image_based_expansion_distance: If true, overrides any set expand_by_pixels value, and expansion
            distance is calculated based on the average label size in loaded region. In this case, expandby_factor must
            be supplied.
        expand_by_factor: Only used if calculate_image_based_expansion_distance is True.
            Multiplier that specifies pixels by which to expand each label. Float in range
            [0, 1 or higher], e.g. 0.2 means that 20% of mean equivalent diameter of labels in region is used.
        fill_holes: if True, the label image prior to expansion has holes filled by iterating
            over slices. Useful for filling lumens in segmentation.
    """

    logger.info(
        f"Running for {zarr_url=}. \n" f"Label expansion for {label_name_to_expand=}."
    )
    # TODO: for NGIO refactor, this task follows logic of surface_mesh_multiscale task
    # TODO: add integration tests

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

    # Read expansion label and ROI table
    label_dask, roi_adata, roi_idlist, label_ngffmeta, label_pixmeta = load_label_rois(
        zarr_url, label_name_to_expand, roi_table, level
    )
    roi_attrs = get_zattrs(f"{zarr_url}/tables/{roi_table}")

    # Check ROI input types
    table_type = roi_attrs["type"]
    if table_type == "masking_roi_table":
        if masking_label_map is None:
            raise ValueError(
                "Masking ROI table selected, but no corresponding label map supplied. "
                "Enter masking_label_map in task input. "
            )

        logger.info(
            f"Using masking ROI table to group input labels by {masking_label_map} segmentation with"
            f"ROI table {roi_table}."
        )

        # ROIs to iterate over
        instance_key = roi_attrs["instance_key"]  # e.g. "label"

        # NGIO FIX, TEMP
        # Check that ROI_table.obs has the right column and extract label_value
        if instance_key not in roi_adata.obs.columns:
            if roi_adata.obs.index.name == instance_key:
                # Workaround for new ngio table
                roi_adata.obs[instance_key] = roi_adata.obs.index
            else:
                raise ValueError(
                    f"In _preprocess_input, {instance_key=} "
                    f" missing in {roi_adata.obs.columns=}"
                )

        roi_labels = roi_adata.obs_vector(instance_key)

    elif table_type == "roi_table":
        logger.info(
            f"Using ROI table {roi_table} for loading objects without masking. "
        )
    else:
        raise ValueError(f"Unrecognized table type {table_type}.")

    # Initialize parameters to save the newly calculated label map
    # Save with same dimensions as child labels from which they are calculated

    if new_label_name is None:
        output_label_name = f"{label_name_to_expand}_expanded"
    else:
        output_label_name = new_label_name

    output_roi_table_name = f"{output_label_name}_ROI_table"

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
        f"New array saved as {output_label_name=} will have shape {shape} and chunks {chunks}."
    )

    # initialize new ROI table
    bbox_dataframe_list = []

    ##############
    # Optionally load parent mask to filter children by parent ###
    ##############

    if masking_label_map:
        # Load masking object segmentation to mask child objects
        # Load well image as dask array for parent objects

        (
            mask_dask,
            mask_adata,
            mask_idlist,
            mask_ngffmeta,
            mask_pixmeta,
        ) = load_label_rois(
            zarr_url,
            masking_label_map,
            roi_table,
            level,
        )

    ##############
    # Apply expansion ###
    ##############

    compute = True

    total_label_count = len(roi_adata.obs_names)

    logger.info(
        f"Starting iteration over {total_label_count} detected objects in ROI table."
    )

    # For each object in input ROI table...
    for i, obsname in enumerate(roi_adata.obs_names):

        if table_type == "masking_roi_table":
            label_str = roi_labels[i]
            region = convert_indices_to_regions(roi_idlist[i])

        elif table_type == "roi_table":
            label_str = obsname
            region = convert_indices_to_regions(roi_idlist[i])

        # Load label image of object to expand as numpy array
        seg = load_region(
            data_zyx=label_dask,
            region=region,
            compute=compute,
        )

        if table_type == "masking_roi_table":
            # Mask objects by parent group_by object
            seg, parent_mask = mask_by_parent_object(
                seg, mask_dask, mask_idlist, i, label_str
            )

        ##############
        # Perform label expansion  ###
        ##############

        # Fill holes, e.g. lumen
        if fill_holes:
            # fill holes in label image
            seg = fill_holes_by_slice_multi_instance(seg)

        if calculate_image_based_expansion_distance:
            expandby = expand_by_factor
        else:
            expandby = expand_by_pixels

        seg_expanded, distance = run_expansion(
            seg,
            expandby,
            expansion_distance_image_based=calculate_image_based_expansion_distance,
        )

        if mask_output and table_type == "masking_roi_table":
            seg_expanded = seg_expanded * parent_mask

        origin_zyx = tuple(s.start for s in region)

        bbox_df = array_to_bounding_box_table(
            seg_expanded,
            label_pixmeta,
            origin_zyx=origin_zyx,
        )

        bbox_dataframe_list.append(bbox_df)

        logger.info(f"Expanded label(s) in region {label_str} by {distance} pixels.")

        ##############
        # Save labels ###
        ##############

        # Store labels as new label map in zarr
        # IF mask_output=False and expanded labels extend beyond parent label, note that pixels of overlap between
        # children of neighboring parents will be overwritten by the last written object.

        save_new_label_with_overlap(
            seg_expanded,
            new_label3d_array,
            zarr_url,
            output_label_name,
            region,
            compute,
        )

    ##############
    # Build pyramid and save new masking ROI table of expanded labels ###
    ##############
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

    bbox_table = save_masking_roi_table_from_df_list(
        bbox_dataframe_list,
        zarr_url,
        output_roi_table_name,
        output_label_name,
        overwrite=True,
    )

    logger.debug(
        pd.DataFrame(
            bbox_table.X,
            index=bbox_table.obs_vector("label"),
            columns=bbox_table.var_names,
        )
    )

    logger.info(f"End expand_labels task for {zarr_url}/labels/{label_name_to_expand}")

    return


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=expand_labels,
        logger_name=logger.name,
    )
