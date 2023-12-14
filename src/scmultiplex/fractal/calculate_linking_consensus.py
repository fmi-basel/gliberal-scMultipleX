# Copyright 2023 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Nicole Repina <nicole.repina@fmi.ch>
# Tommaso Comparin <tommaso.comparin@exact-lab.it>
# Joel Lüthi <joel.luethi@uzh.ch>
#

"""
Calculates consensus for linking tables across multiplexing rounds.
Stores single consensus table in reference round directory.
"""
import logging
from typing import Any
from typing import Sequence

from functools import reduce

import anndata as ad
import numpy as np
import pandas as pd
import zarr

from pydantic.decorator import validate_arguments

from fractal_tasks_core.ngff import load_NgffWellMeta
from fractal_tasks_core.tables import write_table

logger = logging.getLogger(__name__)


@validate_arguments
def calculate_linking_consensus(
    *,
    # Fractal arguments
    input_paths: Sequence[str],
    output_path: str,
    component: str,
    metadata: dict[str, Any],
    # Task-specific arguments
    roi_table: str = "object_linking",
    reference_cycle: int = 0,
) -> dict[str, Any]:
    """
    Applies pre-calculated registration to ROI tables.

    Apply pre-calculated registration such that resulting ROIs contain
    the consensus align region between all cycles.

    Parallelization level: well

    Args:
        input_paths: List of input paths where the image data is stored as
            OME-Zarrs. Should point to the parent folder containing one or many
            OME-Zarr files, not the actual OME-Zarr file. Example:
            `["/some/path/"]`. This task only supports a single input path.
            (standard argument for Fractal tasks, managed by Fractal server).
        output_path: This parameter is not used by this task.
            (standard argument for Fractal tasks, managed by Fractal server).
        component: Path to the OME-Zarr image in the OME-Zarr plate that is
            processed. Example: `"some_plate.zarr/B/03`.
            (standard argument for Fractal tasks, managed by Fractal server).
        metadata: This parameter is not used by this task.
            (standard argument for Fractal tasks, managed by Fractal server).
        roi_table: Name of the ROI table over which the task loops to
            calculate the registration. Examples: `FOV_ROI_table` => loop over
            the field of views, `well_ROI_table` => process the whole well as
            one image.
        reference_cycle: Which cycle to register against. Defaults to 0,
            which is the first OME-Zarr image in the well, usually the first
            cycle that was provided

    """
    # TODO: test this task on sample data with more than two rounds

    consensus_table_name = roi_table + "_consensus"

    logger.info(
        f"Running for {input_paths=}, {component=}. \n"
        f"Applying consensus finding to {roi_table=} and storing it as "
        f"{consensus_table_name=} in reference round {reference_cycle} directory."
    )

    well_zarr_path = f"{input_paths[0]}/{component}"

    well_meta = load_NgffWellMeta(well_zarr_path)
    # dictionary of all rounds present in zarr
    acquisition_dict = well_meta.get_acquisition_paths()
    logger.info(
        "Calculating consensus for the following cycles: "
        f"{acquisition_dict}"
    )

    # TODO: refactor to perform data merging on anndata without need to convert to pandas.
    # Collect all the ROI tables, dictionary key (round id): value (table as anndata)
    roi_tables = {}
    for acq in acquisition_dict.keys():
        # Skip the reference cycle because it will not contain linking table
        if acq == reference_cycle:
            continue

        acq_path = acquisition_dict[acq]
        rx_adata = ad.read_zarr(
            f"{well_zarr_path}/{acq_path}/tables/{roi_table}"
        )

        # Check for valid ROI tables
        are_linking_table_columns_valid(table=rx_adata, reference_cycle=reference_cycle, alignment_cycle=acq)

        # Add to dictionary
        roi_tables[acq] = rx_adata

    # Convert dictionary of anndata tables to list of pandas dfs
    roi_df_list = [
        roi_table.to_df().iloc[:, 0:2] # check jupyter notebook
        for roi_table in roi_tables.values()
    ]

    logger.info("Calculating consensus across cycles.")

    consensus = find_consensus(df_list=roi_df_list, on=["R" + str(reference_cycle) + "_label"])

    consensus = consensus.sort_values(by=["R" + str(reference_cycle) + "_label"])
    consensus['consensus_index'] = consensus.index.astype(np.float32)
    consensus['consensus_label'] = consensus['consensus_index']+1

    # Consensus table has columns ["R0_label", "R1_label", "R2_label", ... "consensus_index", 'consensus_label']
    logger.info(consensus)

    # Convert to adata
    consensus_adata = ad.AnnData(X=np.array(consensus), dtype=np.float32)
    obsnames = list(map(str, consensus.index))
    varnames = list(consensus.columns.values)
    consensus_adata.obs_names = obsnames
    consensus_adata.var_names = varnames

    ##############
    # Storing the calculated consensus ###
    ##############

    # Save the linking table as a new table in reference cycle directory
    logger.info(
        f"Write the consensus ROI table {consensus_table_name} in {reference_cycle} directory"
    )

    image_group = zarr.group(f"{well_zarr_path}/{acquisition_dict[reference_cycle]}")

    write_table(
        image_group,
        consensus_table_name,
        consensus_adata,
        overwrite=True,
        table_attrs=dict(type="linking_table", fractal_table_version="1"),
    )

    return {}


def are_linking_table_columns_valid(*, table: ad.AnnData, reference_cycle: int, alignment_cycle: int) -> None:
    """
    Verify some validity assumptions on a ROI table.

    This function reflects our current working assumptions (e.g. the presence
    of some specific columns); this may change in future versions.

    Args:
        table: AnnData table to be checked
        reference_cycle: reference round id to which all rounds are linked
        alignment_cycle: alignment round id which is being linked to reference
    """
    # Hard constraint: table columns must include some expected ones
    columns = ["R" + str(reference_cycle) + "_label",
               "R" + str(alignment_cycle) + "_label"
    ]
    for column in columns:
        if column not in table.var_names:
            raise ValueError(f"Column {column} is not present in linking table")
    return


def find_consensus(*, df_list: Sequence[pd.DataFrame], on: Sequence[str]) -> pd.DataFrame:
    """
    Find consensus df from a list of dfs where only common ref IDs are kept

    Args:
        df_list: list of dataframes across which consensus is to be found
        on: column name(s) that are in common between rounds
    """

    consensus = reduce(lambda left, right: pd.merge(left, right, on=on, how='outer'), df_list)

    return consensus


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=calculate_linking_consensus,
        logger_name=logger.name,
    )



