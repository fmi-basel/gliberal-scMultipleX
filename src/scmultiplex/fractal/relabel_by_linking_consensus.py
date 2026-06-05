# Copyright 2023 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Nicole Repina <nicole.repina@fmi.ch>
# Tommaso Comparin <tommaso.comparin@exact-lab.it>
# Joel Lüthi <joel.luethi@uzh.ch>
#

"""
Relabels image labels and ROI tables based on consensus linking.
"""
import logging
from pathlib import Path
from typing import Optional

import anndata as ad
import dask.array as da
import numpy as np
import pandas as pd
import zarr
from fractal_tasks_core.labels import prepare_label_group
from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.pyramids import build_pyramid
from fractal_tasks_core.tables import write_table
from fractal_tasks_core.tasks.io_models import InitArgsRegistration
from pydantic import validate_call

from scmultiplex.fractal.fractal_helper_functions import (
    check_for_duplicates,
    correct_label_column,
    extract_acq_info,
    get_zattrs,
    read_table_and_attrs,
)
from scmultiplex.linking.NucleiLinkingFunctions import (
    count_number_of_labels_in_dask,
    make_linking_dict,
    run_relabel_dask,
)

logger = logging.getLogger(__name__)


@validate_call
def relabel_by_linking_consensus(
    *,
    # Fractal arguments
    zarr_url: str,
    init_args: InitArgsRegistration,
    # Task-specific arguments
    label_name: str,
    new_label_name: Optional[str] = None,
    consensus_table: str = "org_match_table_consensus",
    table_to_relabel: str = "org_ROI_table",
    discard_labels_not_linked_across_all_rounds: bool = True,
):
    """
    Relabels image labels and ROI tables based on consensus linking.

    Parallelization level: image

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        init_args: Intialization arguments provided by
            `_image_based_registration_hcs_init`. They contain the
            reference_zarr_url that is used for registration.
            (standard argument for Fractal tasks, managed by Fractal server).
        label_name: Label name to be relabeled; e.g. `org` or `nuc`.
        new_label_name: Optionally new name for relabeled label.
            If left None, default is {label_name}_linked
        consensus_table: Name of consensus matching table that specifies consensus matches across rounds,
            typically stored in reference round zarr.
        table_to_relabel: Table name to relabel based on consensus linking. The table rows correspond
            to specified 'Label name', e.g. 'org_ROI_table' or 'nuc_ROI_table'
        discard_labels_not_linked_across_all_rounds: if True (default), labels that are linked in
            some but not all rounds are discarded, i.e. only objects linked across all rounds
            are kept. If False, partially linked labels are kept (e.g. if label is linked between R0 and R1,
            but missing in R2, the label is still kept).
    """
    # Refactor lines below to make single function for loading?
    # parameter for 'run on reference cycle' true or false; here is True

    # Set OME-Zarr paths
    r0_zarr_path = init_args.reference_zarr_url

    zarr_acquisition = extract_acq_info(zarr_url)
    ref_acquisition = extract_acq_info(r0_zarr_path)

    if new_label_name is None:
        new_label_name = label_name + "_linked"

    new_table_name = new_label_name + "_ROI_table"

    logger.info(
        f"Running for {zarr_url=}. \n"
        f"Relabeling {table_to_relabel=} with labeled objects "
        f"{label_name=} \n"
        f"based on consensus table {consensus_table} in reference round {ref_acquisition} directory."
    )

    ##############
    #  Relabel ROI table
    ##############

    # Read ROIs
    consensus_adata = ad.read_zarr(f"{r0_zarr_path}/tables/{consensus_table}")
    rx_label_adata, table_attrs = read_table_and_attrs(Path(zarr_url), table_to_relabel)
    consensus_pd = consensus_adata.to_df()

    moving_colname = "R" + str(zarr_acquisition) + "_label"
    fixed_colname = "consensus_label"

    # load list of consensus linked rx IDs that have been linked across rounds
    if discard_labels_not_linked_across_all_rounds:
        # drop any row that has NaN in it
        consensus_pd = consensus_pd.dropna()
    else:
        # drop NaN values only of RX round, they would exist if
        # discard_labels_not_linked_across_all_rounds = False and
        # given R0 label does not exist in RX so is NaN in RX only
        consensus_pd = consensus_pd.dropna(subset=[moving_colname])

    # convert floats in linking tables to int then str
    id_rx = consensus_pd[[moving_colname]].copy()  # labels are floats here
    id_list = id_rx[moving_colname].tolist()
    # id_list contains only linked RX objects that want to relabel
    id_list = [str(int(x)) for x in id_list]  # convert to list of strings of integers

    # convert object labels in original ROI table to strings of integers
    # rx_label_adata contains all objects
    rx_label_adata = correct_label_column(rx_label_adata, column_name="label")
    rx_label_adata.obs["label"] = (
        pd.to_numeric(rx_label_adata.obs["label"]).astype(int).astype(str)
    )

    # filter ROI table by rx IDs that have been linked across all rounds i.e. discard non-consensus labels
    # make new ROI table, bdata, that contains only for the linked IDs
    bdata = rx_label_adata[rx_label_adata.obs["label"].isin(id_list)].copy()

    # relabel rx IDs to consensus ID with a matching dictionary
    # matching_dict: key is moving label (rx ID), value is fixed label (consensus ID)
    consensus_pd_str = consensus_pd.astype(int).astype(str)
    matching_dict = make_linking_dict(
        consensus_pd_str, moving_colname=moving_colname, fixed_colname=fixed_colname
    )
    # map original rx IDs to consensus label
    bdata.obs["label"] = bdata.obs["label"].map(matching_dict)
    # reset label index
    bdata.obs.reset_index(drop=True, inplace=True)
    bdata.obs.index = bdata.obs.index.map(str)  # anndata wants indexes as strings!

    # check for duplicated labels after matching
    is_duplicated = check_for_duplicates(bdata.obs["label"])
    if is_duplicated:
        raise ValueError("Detected duplicated labels in output ROI table.")

    # Save the linking table as a new table in round directory
    image_group = zarr.group(f"{zarr_url}")

    # TODO Temporary fix to write correct path to label in table zattr
    table_attrs["region"]["path"] = f"../labels/{new_label_name}"

    write_table(
        image_group,
        new_table_name,
        bdata,
        overwrite=True,
        table_attrs=table_attrs,
    )

    logger.info(
        f"Saved the relabeled ROI table {new_table_name} in round {zarr_acquisition} tables directory"
    )

    ##############
    #  Relabel segmentation image
    ##############

    # 1) Load label image

    # Load label image as dask array, assume that always use level 0 (full res)
    rx_dask = da.from_zarr(f"{zarr_url}/labels/{label_name}/0")
    chunks = rx_dask.chunksize

    # Prepare the output label group
    # Get the label_attrs correctly
    label_attrs = get_zattrs(zarr_url=f"{zarr_url}/labels/{label_name}")

    # useful check for overwriting, adds metadata to labels group
    _ = prepare_label_group(
        image_group=zarr.group(zarr_url),
        label_name=new_label_name,
        overwrite=True,
        label_attrs=label_attrs,
        logger=logger,
    )

    # 2) Relabel image

    # Loop over linked labels and relabel. if label not in consensus, it is set to 0 (background).
    logger.info(f"Relabeling {zarr_url=} image...")

    rx_dask_relabeled = run_relabel_dask(
        label_dask=rx_dask,
        matches=consensus_pd,
        moving_colname=moving_colname,
        fixed_colname=fixed_colname,
    )

    count_input, labels_in_input = count_number_of_labels_in_dask(rx_dask)

    count_output, labels_in_output = count_number_of_labels_in_dask(rx_dask_relabeled)

    # Check outputs
    if count_input != rx_label_adata.n_obs:
        raise ValueError(
            f"Input label image contains {count_input} labels while input ROI table contains {rx_label_adata.n_obs} "
            f"labels. Does input ROI table match input label image?"
        )
    if count_output != bdata.n_obs:
        roi_set = set(bdata.obs["label"].to_numpy().flatten().astype(float))
        raise ValueError(
            f"Label count {count_output} in relabelled image must match length of relabelled table {bdata.n_obs}. "
            f"\nLabels in relabelled image but not in relabelled ROI table: \n{labels_in_output - roi_set}"
            f"\nLabels in relabelled ROI table but not in relabelled image: \n{roi_set - labels_in_output}"
        )

    logger.info(f"Relabeled {count_output} out of {count_input} detected labels")

    # 3) Save changed label image to OME-Zarr

    label_dtype = np.uint32
    # this could be restructured to use output of prepare_label_group, work in progress
    store = zarr.storage.FSStore(f"{zarr_url}/labels/{new_label_name}/0")
    new_label_array = zarr.create(
        shape=rx_dask_relabeled.shape,
        chunks=chunks,
        dtype=label_dtype,
        store=store,
        overwrite=True,
        dimension_separator="/",
    )

    da.array(rx_dask_relabeled).to_zarr(
        url=new_label_array,
    )
    logger.info(f"Saved {new_label_name} to Zarr at full resolution")

    # 4) Build pyramids for label image

    # load meta for original label image just to get metadata
    label_meta = load_NgffImageMeta(f"{zarr_url}/labels/{label_name}")

    build_pyramid(
        zarrurl=f"{zarr_url}/labels/{new_label_name}",
        overwrite=True,
        num_levels=label_meta.num_levels,
        coarsening_xy=label_meta.coarsening_xy,
        chunksize=chunks,
        aggregation_function=np.max,
    )

    logger.info(f"Built a pyramid for the {new_label_name} label image")


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=relabel_by_linking_consensus,
        logger_name=logger.name,
    )
