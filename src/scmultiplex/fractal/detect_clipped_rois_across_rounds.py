# Copyright 2025 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Nicole Repina <nicole.repina@fmi.ch>
#

"""
Detect clipped ROIs across rounds.
"""

import logging
import os

import numpy as np
from fractal_tasks_core.tasks.io_models import InitArgsRegistrationConsensus
from ngio import open_ome_zarr_container
from ngio.utils import ngio_logger
from pydantic import validate_call

# Local application imports
from scmultiplex.fractal.fractal_helper_functions import (
    clear_registration_folder,
    extract_acq_info,
)
from scmultiplex.linking.OrganoidLinkingFunctions import group_by_roi_name_from_dict

# Configure logging
ngio_logger.setLevel("ERROR")
logger = logging.getLogger(__name__)


@validate_call
def detect_clipped_rois_across_rounds(
    *,
    # Fractal arguments
    zarr_url: str,
    init_args: InitArgsRegistrationConsensus,
    # Task-specific arguments
    roi_table_name: str = None,
    save_folder_name: str = None,
    overwrite_folder: bool = False,
):
    """
    Find labels to delete from multiplexing rounds.
    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        init_args: Intialization arguments provided by
            `init_select_all_knowing_reference`. They contain the reference_zarr_url information.
            (standard argument for Fractal tasks, managed by Fractal server).
        roi_table_name: Name of masking ROI table to be used for ROI loading.
        save_folder_name: Name of folder where to save the labels to be deleted. Saved in reference directory.
        overwrite_folder: If True, clear existing subfolder {save_folder_name} in 'registration' folder to allow
            re-run of task with same output name.

    """

    logger.info(f"Start detect_clipped_rois_across_rounds for {zarr_url=}")

    # use init_select_reference_knowing_all.py
    # the zarr_url is the reference zarr

    # Identify labels to remove, which have different ROI shapes across rounds:

    # Open the ome-zarr image container
    # Perform all lazily with dask
    ome_zarr = open_ome_zarr_container(zarr_url)
    roi_table = ome_zarr.get_table(roi_table_name, check_type="generic_roi_table")

    ref_acquisition = extract_acq_info(zarr_url)

    # Collect ROI tables for all rounds (including reference), dictionary key (round id): value (table as anndata)
    roi_rois = {}
    roi_names = {}
    for acq_zarr_url in init_args.zarr_url_list:
        round_id = extract_acq_info(acq_zarr_url)

        ome_zarr = open_ome_zarr_container(acq_zarr_url)
        roi_table = ome_zarr.get_table(roi_table_name, check_type="generic_roi_table")

        roi_name_list = []
        for roi in roi_table.rois():
            # extract names
            roi_name_list.append(roi.name)

        roi_rois[round_id] = roi_table.rois()
        roi_names[round_id] = roi_name_list

    # Convert all dict values to sets
    sets = [set(v) for v in roi_names.values()]

    # Compute intersection across all sets to find common values
    common_values = set.intersection(*sets)

    # Find ROI IDs missing in at least 1 moving round
    reference_values = set(roi_names[ref_acquisition])
    rois_missing_in_at_least_1_moving_round = reference_values.difference(
        common_values
    )  # set

    logger.info(
        f"Detected {len(rois_missing_in_at_least_1_moving_round)} ROIs missing from at least one "
        f"moving round."
    )

    if len(rois_missing_in_at_least_1_moving_round) > 0:
        logger.info(f"Missing ROIs: {rois_missing_in_at_least_1_moving_round}.")

    roi_groups = group_by_roi_name_from_dict(roi_rois)

    rois_clipped = set()

    for ref_roi in roi_rois[ref_acquisition]:

        # Skip labels that will be deleted anyway
        if ref_roi.name in rois_missing_in_at_least_1_moving_round:
            continue

        fields_to_check = ["x_length", "y_length", "z_length"]

        for field in fields_to_check:
            reference_value = getattr(ref_roi, field)
            list_of_all_rois_all_rounds = roi_groups[ref_roi.name]
            for r in list_of_all_rois_all_rounds:
                if getattr(r, field) != reference_value:
                    rois_clipped.add(ref_roi.name)

    logger.info(f"Detected {len(rois_clipped)} clipped ROIs.")

    if len(rois_clipped) > 0:
        logger.info(f"Clipped ROIs: {rois_clipped}.")

    # Set save location and clear if necessary
    registration_save_path = os.path.join(zarr_url, "registration", save_folder_name)
    if os.path.isdir(registration_save_path):
        if overwrite_folder:
            clear_registration_folder(save_folder_name, zarr_url)
            logger.info(f"Cleared existing registration folder: {save_folder_name=}")
        else:
            raise ValueError(
                f"Folder {save_folder_name=} already exists. To overwrite, set "
                f"overwrite_folder=True."
            )
    else:
        # Make directory
        os.makedirs(registration_save_path)

    roi_to_delete = rois_clipped.union(rois_missing_in_at_least_1_moving_round)
    # Convert set to numpy array
    labels_to_delete = np.array(list(roi_to_delete), dtype=int)
    # Save computed warp map as numpy .npz file
    filename = "labels_to_delete.npz"
    np_save_path = os.path.join(registration_save_path, filename)

    np.savez_compressed(
        np_save_path,
        labels_to_delete=labels_to_delete,
    )

    logger.info(f"Saved labels to delete as numpy array .npz in {np_save_path}")

    logger.info(f"End detect_clipped_rois_across_rounds for {zarr_url=}")

    return {}


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=detect_clipped_rois_across_rounds,
        logger_name=logger.name,
    )
