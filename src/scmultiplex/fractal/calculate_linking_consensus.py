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
from typing import Optional

import anndata as ad
import numpy as np
import zarr
from fractal_tasks_core.tables import write_table
from fractal_tasks_core.tasks.io_models import InitArgsRegistrationConsensus
from pydantic import validate_call

from scmultiplex.fractal.fractal_helper_functions import (
    are_linking_table_columns_valid,
    extract_acq_info,
    find_consensus,
)

logger = logging.getLogger(__name__)


@validate_call
def calculate_linking_consensus(
    *,
    # Fractal arguments
    zarr_url: str,
    init_args: InitArgsRegistrationConsensus,
    # Task-specific arguments
    roi_table: str = "org_match_table",
    new_consensus_table_name: Optional[str] = None,
):
    """
    Applies pre-calculated registration to ROI tables.

    Apply pre-calculated registration such that resulting ROIs contain
    the consensus align region between all cycles.

    Parallelization level: well

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            Refers to the zarr_url of the reference acquisition.
            (standard argument for Fractal tasks, managed by Fractal server).
        init_args: Intialization arguments provided by
            `init_group_by_well_for_multiplexing`. It contains the
            zarr_url_list listing all the zarr_urls in the same well as the
            zarr_url of the reference acquisition that are being processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        roi_table: Name of the matching table used to calculate consensus across
            multiplexing rounds. Typically, this matching table is the output of a linking task,
            e.g. org_match_table or nuc_match_table. This table must be present in all rounds
            except for reference round.
        new_consensus_table_name: Optional new table name for consensus matches, saved in reference round directory.
            If left None, default is {roi_table} + "_consensus"

    """

    if new_consensus_table_name is None:
        consensus_table_name = roi_table + "_consensus"
    else:
        consensus_table_name = new_consensus_table_name

    # zarr_url is the path to the reference round
    # init args contain the list to all rounds, including reference round
    ref_acquisition = extract_acq_info(zarr_url)

    logger.info(
        f"Running for reference round {zarr_url=}. \n"
        f"Applying consensus finding to {roi_table=} and storing it as "
        f"{consensus_table_name=} in reference round {ref_acquisition} directory."
    )

    # TODO: refactor to perform data merging on anndata without need to convert to pandas.
    # Collect all the ROI tables, dictionary key (round id): value (table as anndata)
    roi_tables = {}

    for acq_zarr_url in init_args.zarr_url_list:
        zarr_acquisition = extract_acq_info(acq_zarr_url)
        # Skip the reference cycle because it will not contain linking table
        if acq_zarr_url == zarr_url:
            continue

        rx_adata = ad.read_zarr(f"{acq_zarr_url}/tables/{roi_table}")

        # Check for valid ROI tables
        are_linking_table_columns_valid(
            table=rx_adata,
            reference_cycle=ref_acquisition,
            alignment_cycle=zarr_acquisition,
        )

        # Add to dictionary
        roi_tables[zarr_acquisition] = rx_adata

    # Convert dictionary of anndata tables to list of pandas dfs
    roi_df_list = [
        roi_table.to_df().iloc[:, 0:2]  # check jupyter notebook
        for roi_table in roi_tables.values()
    ]

    logger.info("Calculating consensus across cycles.")

    consensus = find_consensus(
        df_list=roi_df_list, on=["R" + str(ref_acquisition) + "_label"]
    )

    consensus = consensus.sort_values(by=["R" + str(ref_acquisition) + "_label"])
    consensus["consensus_index"] = consensus.index.astype(np.float32)
    consensus["consensus_label"] = consensus["consensus_index"] + 1

    # Consensus table has columns ["R0_label", "R1_label", "R2_label", ... "consensus_index", 'consensus_label']
    logger.info(
        f"Successfully calculated consensus labels for {len(consensus.index)} objects across round(s) "
        f"{list(roi_tables.keys())} and reference round {ref_acquisition}."
    )

    # Convert to adata
    consensus_adata = ad.AnnData(X=np.array(consensus, dtype=np.float32))
    obsnames = list(map(str, consensus.index))
    varnames = list(consensus.columns.values)
    consensus_adata.obs_names = obsnames
    consensus_adata.var_names = varnames

    ##############
    # Storing the calculated consensus ###
    ##############

    # Save the linking table as a new table in reference cycle directory
    logger.info(
        f"Write the consensus ROI table {consensus_table_name} in {ref_acquisition} directory"
    )

    image_group = zarr.group(f"{zarr_url}")

    write_table(
        image_group,
        consensus_table_name,
        consensus_adata,
        overwrite=True,
        table_attrs=dict(type="linking_table", fractal_table_version="1"),
    )


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=calculate_linking_consensus,
        logger_name=logger.name,
    )
