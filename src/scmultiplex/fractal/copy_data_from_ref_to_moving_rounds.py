# Copyright 2026 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Nicole Repina <nicole.repina@fmi.ch>
#

import logging
import os
from pathlib import Path
from typing import Optional

from fractal_tasks_core.tasks.io_models import InitArgsRegistration
from ngio import open_ome_zarr_container
from ngio.utils import ngio_logger
from pydantic import validate_call

from scmultiplex.utils.ngio_utils import get_acquisition_id

# Configure logging
ngio_logger.setLevel("ERROR")
logger = logging.getLogger("copy_data_from_ref_to_moving_rounds")


@validate_call
def copy_data_from_ref_to_moving_rounds(
    *,
    # Fractal arguments
    zarr_url: str,
    init_args: InitArgsRegistration,
    # Task-specific arguments
    roi_table_names_to_copy_from_ref: Optional[list[str]] = None,
    overwrite: bool = False,
):
    """
    Copy data (e.g. tables, corresponding label images) from reference to moving multiplexing rounds.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        init_args: Intialization arguments provided by
            `init_select_all_knowing_reference`. They contain the reference_zarr_url information.
            (standard argument for Fractal tasks, managed by Fractal server).
        roi_table_names_to_copy_from_ref: List of ROI table names present in reference round to copy
            over. The corresponding label of masking ROI tables are also copied over.
        overwrite: If True, overwrite labels and tables with same name in registered zarr image.

    """
    # iterate over each round, not pairwise loading
    reference_zarr_url = init_args.reference_zarr_url

    # Get id of current moving acquisition from plate/well metadata as int, e.g. 1
    reference_acquisition_id = get_acquisition_id(reference_zarr_url)
    current_acquisition_id = get_acquisition_id(zarr_url)

    # skip the reference round
    if current_acquisition_id == reference_acquisition_id:
        return

    # Open the ome-zarr image container from which to copy
    source_ome_zarr = open_ome_zarr_container(reference_zarr_url)
    target_zarr_image_name = os.path.basename(zarr_url)
    target_ome_zarr = open_ome_zarr_container(zarr_url)

    if roi_table_names_to_copy_from_ref is not None:
        label_names_to_copy = []

        # Copy tables
        for table_name in roi_table_names_to_copy_from_ref:
            table = source_ome_zarr.get_table(table_name)

            logger.info(
                f"Copying table {table_name} from registered reference round to "
                f"{target_zarr_image_name}."
            )

            target_ome_zarr.add_table(table_name, table, overwrite=overwrite)
            # Make list of labels to copy
            if table.type() == "masking_roi_table":
                # get name of corresponding label as stored in table metadata
                label_names_to_copy.append(Path(table._meta.region.path).name)

        # Copy labels
        for i, label_name in enumerate(label_names_to_copy):
            logger.info(
                f"Copying label {label_name} from registered reference round to "
                f"{target_zarr_image_name} as {label_name}."
            )

            label_image = source_ome_zarr.get_label(name=label_name)
            label_image_dask = label_image.get_array(mode="dask")
            new_label = target_ome_zarr.derive_label(
                label_name,
                dtype=label_image_dask.dtype,
                overwrite=overwrite,
            )
            new_label.set_array(label_image_dask)
            new_label.consolidate()

    return


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=copy_data_from_ref_to_moving_rounds)
