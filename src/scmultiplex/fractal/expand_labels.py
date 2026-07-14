# Copyright (C) 2024 Friedrich Miescher Institute for Biomedical Research

##############################################################################
#                                                                            #
# Author: Nicole Repina              <nicole.repina@fmi.ch>                  #
#                                                                            #
##############################################################################
import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
from ngio import open_ome_zarr_container
from ngio.utils import ngio_logger
from pydantic import validate_call

from scmultiplex.fractal.fractal_helper_functions import (
    roi_to_pixel_slices,
    save_new_label_image_with_overlap,
)
from scmultiplex.meshing.LabelFusionFunctions import (
    fill_holes_by_slice_multi_instance,
    run_expansion,
)

ngio_logger.setLevel("ERROR")
logger = logging.getLogger("expand_labels")


@validate_call
def expand_labels(
    # Fractal arguments
    zarr_url: str,
    # Task-specific arguments
    label_name_to_expand: str,
    new_label_name: Optional[str] = None,
    parent_roi_table_name: str = "",
    mask_input: bool = False,
    mask_output: bool = False,
    expand_by_pixels: Union[int, None] = None,
    calculate_image_based_expansion_distance: bool = False,
    expand_by_factor: Union[float, None] = None,
    fill_holes: bool = False,
    expand_in_z: bool = False,
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
        parent_roi_table_name: Name of the ROI table used to iterate over objects and load object regions. If a table of type
            "roi_table" is passed, e.g. well_ROI_table, all objects for each region in the table will be loaded
            and expanded simultaneously. If a table of type "masking_roi_table" is passed, e.g. a segmentation
            ROI table, the task iterates over these objects and loads only the children (i.e. label_name_to_expand) that
            belong to the parent object.
        mask_input: If True, input non-expanded label is masked by parent label prior to expansion. If True, this
            selects children belonging to a given parent. However, this may crop input in cases where parent mask
            segmentation is not perfect.
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
        expand_in_z: if True, uses different 3d expansion function to expand isotropically in z,y,x. Use with caution -
            anisotropic voxel spacing is not supported.
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

    ome_zarr = open_ome_zarr_container(zarr_url)

    # Load ROI tables
    parent_roi_table = ome_zarr.get_table(
        parent_roi_table_name, check_type="generic_roi_table"
    )

    table_type = Path(parent_roi_table._meta.type).name
    # Set masking to True if input is a masking_roi_table
    masking = table_type == "masking_roi_table"

    if masking:
        logger.info("Masking ROI table detected.")
        parent_label_name = Path(parent_roi_table._meta.region.path).name

        # Load label image to expand
        label_image = ome_zarr.get_masked_label(
            label_name=label_name_to_expand, masking_label_name=parent_label_name
        )
        if mask_output:
            parent_masked_image = ome_zarr.get_masked_label(
                label_name=parent_label_name, masking_label_name=parent_label_name
            )
    else:
        logger.info("No masking ROI table detected.")
        label_image = ome_zarr.get_label(name=label_name_to_expand)

    # Pixel sizes as list: [z,y,x]
    pixel_size = label_image.pixel_size
    spacing = np.array([pixel_size.z, pixel_size.y, pixel_size.x])

    # Initialize parameters to save the newly calculated label map
    if new_label_name is None:
        output_label_name = f"{label_name_to_expand}_expanded"
    else:
        output_label_name = new_label_name

    output_roi_table_name = f"{output_label_name}_ROI_table"

    # Initialize zarr with NGIO
    ome_zarr.derive_label(name=output_label_name, ref_image=label_image, overwrite=True)

    ##############
    # Apply expansion ###
    ##############

    logger.info(
        f"Starting label expansion for {len(parent_roi_table.rois())} detected ROIs..."
    )

    # For each object in input ROI table...
    for roi in parent_roi_table.rois():
        label_string = roi.name
        logger.info(f"Processing ROI label name '{label_string}'")

        # Load numpy of segmentation images
        if masking and mask_input:
            label_int = int(label_string)
            seg = label_image.get_roi_masked(label=label_int)  # z,y,x
        elif masking:
            label_int = int(label_string)
            seg = label_image.get_roi(label=label_int)  # do not mask by parent
        else:
            seg = label_image.get_roi(roi=roi)  # e.g. well or FoV ROI table

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
            expand_in_z=expand_in_z,
        )

        if masking and mask_output:
            parent_mask = parent_masked_image.get_roi_masked(label=label_int)

            parent_mask[parent_mask > 0] = 1  # Binarize

            seg_expanded = seg_expanded * parent_mask

        logger.info(f"Expanded label(s) in region {label_string} by {distance} pixels.")

        ##############
        # Save labels ###
        ##############
        # Save ROI to disk using dask _to_zarr, not ngio
        region = roi_to_pixel_slices(roi, spacing)
        save_new_label_image_with_overlap(
            seg_expanded, zarr_url, output_label_name, region
        )

    ##############
    # Build pyramid and save new masking ROI table of expanded labels ###
    ##############

    # Build pyramids
    expanded_label_img = ome_zarr.get_label(name=output_label_name)
    expanded_label_img.consolidate()
    logger.info(f"Built pyramid for the {output_label_name} label image")

    # Make ROI table from expanded label image
    masking_table = ome_zarr.build_masking_roi_table(output_label_name)
    ome_zarr.add_table(output_roi_table_name, masking_table, overwrite=True)
    logger.info(f"Saved new masking ROI table as {output_roi_table_name}")

    logger.info(f"End expand_labels task for {zarr_url}/labels/{label_name_to_expand}")

    return


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=expand_labels)
