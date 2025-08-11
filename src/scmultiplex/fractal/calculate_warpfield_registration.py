# Copyright 2025 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Nicole Repina <nicole.repina@fmi.ch>
#

"""
Calculate warpfield registration for multiplexed object pairs and save warp map as npz file.
"""

import copy
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
from fractal_tasks_core.channels import ChannelInputModel
from fractal_tasks_core.tasks.io_models import InitArgsRegistration
from ngio import open_ome_zarr_container
from ngio.utils import ngio_logger
from pydantic import validate_call

# Local application imports
from scmultiplex.fractal.fractal_helper_functions import (
    clear_registration_folder,
    get_channel_index_from_inputmodel,
)

# Configure logging
ngio_logger.setLevel("ERROR")
logger = logging.getLogger(__name__)


@validate_call
def calculate_warpfield_registration(
    *,
    # Fractal arguments
    zarr_url: str,
    init_args: InitArgsRegistration,
    # Task-specific arguments
    registration_channel: ChannelInputModel,
    roi_table_name: str,
    path_to_warpfield_recipe: Optional[str] = None,
    warpfield_pre_filter_clip_thresh: Optional[int] = None,
    registration_name: str = "warpfield",
    overwrite_folder: bool = False,
):
    """
    Calculate warpfield registration between multiplexing rounds for object pairs. Reference and moving
    objects are assumed to be already linked to have matching label ids and roi shape. Pairs of ROIs
    are loaded in 3D and registered using Warpfield (GPU-based 3D non-rigid volumetric registration). This
    task only calculates the warp map, which is stored on disk to then be applied to multichannel images
    using the apply_warpfield_registration task.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        init_args: Intialization arguments provided by
            `_image_based_registration_hcs_init`. They contain the
            reference_zarr_url that is used for registration.
            (standard argument for Fractal tasks, managed by Fractal server).
        registration_channel: Image channel to be used for registration. Must be present in both
            reference and moving rounds.
        roi_table_name: Name of masking ROI table to be used for loading region pairs between rounds. The
            table must be present in both reference and moving rounds and correspond to segmented
            objects that have been linked (e.g. with Calculate Object Linking and Shift by Shift tasks). Each ref/mov
            object pair is expected to have matching label value and shape (bounding box dimension).
        path_to_warpfield_recipe: Absolute path to warpfield recipe file. If None, default.yml is used.
        warpfield_pre_filter_clip_thresh: Optional modifier of warpfield recipe. Pixel value threshold for
            clipping each volume. The default value in default.yml is 0. Higher values remove DC background.
        registration_name: Name of folder that contains warp map .npz files per moving object. Created as
            subfolder of 'registration' folder.
        overwrite_folder: If True, clear existing subfolder {registration_name} in 'registration' folder to allow
            re-run of registration with same name.
    """
    try:
        import warpfield
    except ImportError as e:
        raise ImportError(
            "The `calculate_warpfield_registration` task requires GPU. "
        ) from e

    logger.info(f"Running 'calculate_warpfield_registration' task for {zarr_url=}.")

    # Set OME-Zarr paths
    reference_zarr_url = init_args.reference_zarr_url

    # Open the ome-zarr image container
    reference_ome_zarr = open_ome_zarr_container(reference_zarr_url)
    moving_ome_zarr = open_ome_zarr_container(zarr_url)

    # Load ROI tables
    reference_roi_table = reference_ome_zarr.get_table(
        roi_table_name, check_type="generic_roi_table"
    )
    moving_roi_table = moving_ome_zarr.get_table(
        roi_table_name, check_type="generic_roi_table"
    )
    reference_label_name = Path(reference_roi_table._meta.region.path).name
    moving_label_name = Path(moving_roi_table._meta.region.path).name

    # Load images from container
    reference_image = reference_ome_zarr.get_masked_image(
        masking_label_name=reference_label_name
    )
    moving_image = moving_ome_zarr.get_masked_image(
        masking_label_name=moving_label_name
    )

    # Get channel indeces
    channel_index_ref = get_channel_index_from_inputmodel(
        reference_zarr_url, registration_channel
    )
    channel_index_move = get_channel_index_from_inputmodel(
        zarr_url, registration_channel
    )

    # Set warp map save location and clear if necessary
    registration_save_path = os.path.join(zarr_url, "registration", registration_name)
    if os.path.isdir(registration_save_path):
        if overwrite_folder:
            clear_registration_folder(registration_name, zarr_url)
            logger.info(f"Cleared existing registration folder: {registration_name=}")
        else:
            raise ValueError(
                f"Folder {registration_name=} already exists. To overwrite, set "
                f"overwrite_folder=True."
            )
    else:
        # Make directory
        os.makedirs(registration_save_path)

    # Get warpfield recipe
    if path_to_warpfield_recipe is not None:
        recipe = warpfield.Recipe.from_yaml(path_to_warpfield_recipe)
    else:
        recipe = warpfield.Recipe.from_yaml("default.yml")
    if warpfield_pre_filter_clip_thresh is not None:
        recipe.pre_filter.clip_thresh = warpfield_pre_filter_clip_thresh

    # Log warpfield recipe
    print_recipe = (
        f"Applying recipe...\n"
        f"\nPrefilter:\n{recipe.pre_filter}\n\n"
        + "\n\n".join(f"Level {i}:\n{level}" for i, level in enumerate(recipe.levels))
    )
    logger.info(print_recipe)

    # Calculate warpfield correction
    for roi in reference_roi_table.rois():
        # Deep copy recipe to isolate any modifications per iteration
        recipe_copy = copy.deepcopy(recipe)

        label_string = roi.name
        label_int = int(label_string)
        logger.info(f"Processing ROI label {label_string}")
        # Load matching ROIs (matching label value)
        reference_np = reference_image.get_roi(label=label_int, c=channel_index_ref)

        try:
            moving_np = moving_image.get_roi(label=label_int, c=channel_index_move)
        except KeyError as e:
            logger.warning(
                f"Moving image does not contain matching ROI. Skipping reference ROI {roi}. Error: {e}"
            )
            continue

        reference_np = reference_np.squeeze(axis=0)  # Remove the channel dimension
        moving_np = moving_np.squeeze(axis=0)

        moving_shape = moving_np.shape

        if reference_np.shape != moving_shape:
            raise ValueError(
                f"Reference ROI shape {reference_np.shape} does not match moving ROI "
                f"shape {moving_shape}. Check input ROI table or pre-process ROIs to have "
                f"matching shapes for each region pair."
            )

        logger.info(
            f"Loaded matching ROI pairs with ref shape: {reference_np.shape}, mov shapes: {moving_shape}"
        )

        # Check that blocksize is not larger than image shape. If it is, reduce blocksize to match shape.
        # Otherwise get ValueError: C2R/R2C PlanNd for F-order arrays is not supported
        # Loop over each registration level
        for i, level in enumerate(recipe_copy.levels):
            blocksize = list(level.block_size)
            original_blocksize = blocksize.copy()

            # Adjust blocksize only where it's too large
            for dim, (s, b) in enumerate(zip(list(moving_shape), blocksize)):
                if s < b:
                    blocksize[dim] = s

            # Only apply change if blocksize was modified
            if blocksize != original_blocksize:
                logger.warning(
                    f"Blocksize {original_blocksize} too large for ROI of shape {moving_shape}. "
                    f"Decreased blocksize of level {i} to {blocksize}."
                )
                recipe_copy.levels[i].block_size = blocksize

        # Perform warpfield registration
        _, warp_map, _ = warpfield.register_volumes(
            reference_np, moving_np, recipe_copy
        )

        # Save computed warp map as numpy .npz file
        filename = f"{label_string}.npz"
        warp_map_save_path = os.path.join(registration_save_path, filename)

        np.savez_compressed(
            warp_map_save_path,
            mov_shape=warp_map.mov_shape,
            ref_shape=warp_map.ref_shape,
            block_size=warp_map.block_size,
            block_stride=warp_map.block_stride,
            warp_field=warp_map.warp_field,
        )

    logger.info(f"End calculate_warpfield_registration task for {zarr_url=}")

    return {}


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=calculate_warpfield_registration,
        logger_name=logger.name,
    )
