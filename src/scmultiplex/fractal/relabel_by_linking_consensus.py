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
from typing import Any

import anndata as ad
import dask.array as da
import numpy as np
import zarr
from fractal_tasks_core.labels import prepare_label_group
from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.pyramids import build_pyramid
from fractal_tasks_core.tables import write_table
from fractal_tasks_core.tasks.io_models import InitArgsRegistration
from pydantic.decorator import validate_arguments

from scmultiplex.fractal.fractal_helper_functions import get_zattrs, read_table_and_attrs, extract_acq_info
from scmultiplex.linking.NucleiLinkingFunctions import relabel_RX_numpy, make_linking_dict

logger = logging.getLogger(__name__)


@validate_arguments
def relabel_by_linking_consensus(
        *,
        # Fractal arguments
        zarr_url: str,
        init_args: InitArgsRegistration,
        # Task-specific arguments
        label_name: str,
        roi_table: str = "well_ROI_table",
        consensus_table: str = "org_match_table_consensus",
        table_to_relabel: str = "org_ROI_table",
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
        roi_table: Name of the ROI table that is parent of label_name. For example segmented
            organoids or nuclei labels are usually unique across well and so `well_ROI_table` is used.
        consensus_table: Name of ROI table that contains consensus linking for label_name across all rounds
        table_to_relabel: Table name to relabel based on consensus linking. The table rows correspond
            to label_name, e.g. 'org_ROI_table' or 'nuc_ROI_table'
    """
    # Refactor lines below to make single function for loading?
    # parameter for 'run on reference cycle' true or false; here is True

    # Set OME-Zarr paths
    r0_zarr_path = init_args.reference_zarr_url

    zarr_acquisition = extract_acq_info(zarr_url)
    ref_acquisition = extract_acq_info(r0_zarr_path)

    new_label_name = label_name + '_linked'
    new_table_name = table_to_relabel + '_linked'

    logger.info(
        f"Running for {zarr_url=}. \n"
        f"Relabeling {table_to_relabel=} with labeled objects "
        f"{label_name=} for each region in {roi_table=} \n"
        f"based on consensus table {consensus_table} in reference round {ref_acquisition} directory."
    )

    ##############
    #  Relabel ROI table
    ##############

    # Read ROIs
    consensus_adata = ad.read_zarr(f"{r0_zarr_path}/tables/{consensus_table}")
    rx_label_adata, table_attrs = read_table_and_attrs(
        Path(zarr_url),
        table_to_relabel
    )
    consensus_pd = consensus_adata.to_df()

    moving_colname = "R" + str(zarr_acquisition) + "_label"
    fixed_colname = "consensus_label"

    # make list of rx IDs (list of strings) that are linked across all rounds
    # note that label IDs in ROI tables are strings, so need to convert floats in linking tables to int then str
    id_rx = consensus_adata[:, [moving_colname]].to_df()
    id_list = id_rx[moving_colname].tolist()
    id_list = [str(int(x)) for x in id_list]

    # filter ROI table by rx IDs that have been linked across all rounds i.e. discard non-consensus labels
    # make new table only for the linked IDs
    bdata = rx_label_adata[rx_label_adata.obs['label'].isin(id_list)].copy()

    # relabel rx IDs to consensus ID with a mapping dictionary
    # key is moving label (rx ID), value is fixed label (consensus ID)
    consensus_pd_str = consensus_pd.astype(int).astype(str)
    matching_dict = make_linking_dict(consensus_pd_str, moving_colname=moving_colname, fixed_colname=fixed_colname)
    # map original rx IDs to consensus label
    bdata.obs['label'] = bdata.obs['label'].map(matching_dict)
    # reset label index
    bdata.obs.reset_index(drop=True, inplace=True)
    bdata.obs.index = bdata.obs.index.map(str) # anndata wants indexes as trings!

    # logger.info(bdata.obs)

    # Save the linking table as a new table in round directory
    image_group = zarr.group(f"{zarr_url}")

    # TODO Temporary fix to write correct path to label in table zattr
    table_attrs['region']['path'] = f"../labels/{new_label_name}"

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
    logger.info(
        f"Relabeling {zarr_url=} image..."
    )

    rx_dask_relabeled, count_input, count_output = relabel_RX_numpy(rx_dask, consensus_pd,
                                                                    moving_colname=moving_colname,
                                                                    fixed_colname=fixed_colname, daskarr=True)

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

    if count_output != bdata.n_obs:
        raise ValueError(
            "Label count in relabelled image must match length of relabelled table"
        )


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=relabel_by_linking_consensus,
        logger_name=logger.name,
    )

