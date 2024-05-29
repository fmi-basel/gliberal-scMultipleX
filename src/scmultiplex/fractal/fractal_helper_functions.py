# Copyright 2024 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Nicole Repina <nicole.repina@fmi.ch>
# Tommaso Comparin <tommaso.comparin@exact-lab.it>
# Joel Lüthi <joel.luethi@uzh.ch>
#
from pathlib import Path
import anndata as ad
import zarr
import pandas as pd
import numpy as np
from fractal_tasks_core.ngff import load_NgffWellMeta
from fractal_tasks_core.roi import empty_bounding_box_table
from functools import reduce
from typing import Sequence


def read_table_and_attrs(zarr_url: Path, roi_table):
    table_url = zarr_url / f"tables/{roi_table}"
    table = ad.read_zarr(table_url)
    table_attrs = get_zattrs(table_url)
    return table, table_attrs

# TODO update relabel_by_linking_consensus task to use these functions
def get_zattrs(zarr_url):
    with zarr.open(zarr_url, mode="r") as zarr_img:
        return zarr_img.attrs.asdict()


def convert_indices_to_origin_zyx(
    index: list[int],
) -> tuple[int, int, int]:
    """
    Converts index tuples to origin zyx tuple

    Args:
        index: Tuple containing 6 entries of (z_start, z_end, y_start,
            y_end, x_start, x_end).

    Returns:
        region: tuple of 3 integers (z_start, y_start, x_start)
    """
    return index[0], index[2], index[4]


def format_roi_table(bbox_dataframe_list):
    """
    Formats ROI table to anndata
    Copied from cellpose Fractal task
    Returns anndata to save
    """
    # Handle the case where `bbox_dataframe_list` is empty (typically
    # because list_indices is also empty)
    if len(bbox_dataframe_list) == 0:
        bbox_dataframe_list = [empty_bounding_box_table()]
    # Concatenate all ROI dataframes
    df_well = pd.concat(bbox_dataframe_list, axis=0, ignore_index=True)
    df_well.index = df_well.index.astype(str)
    # Extract labels and drop them from df_well
    labels = pd.DataFrame(df_well["label"].astype(str))
    df_well.drop(labels=["label"], axis=1, inplace=True)
    # Convert all to float (warning: some would be int, in principle)
    bbox_dtype = np.float32
    df_well = df_well.astype(bbox_dtype)
    # Convert to anndata
    bbox_table = ad.AnnData(df_well, dtype=bbox_dtype)
    bbox_table.obs = labels

    return bbox_table


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


def extract_acq_info(zarr_url, ref_url):
    zarr_acquisition = None
    ref_acquisition = None

    zarr_pathname = Path(zarr_url).name
    ref_pathname = Path(ref_url).name
    wellmeta = load_NgffWellMeta(str(Path(zarr_url).parent)).well.images #list of dictionaries for each round
    for img in wellmeta:
        if img.path == zarr_pathname:
            zarr_acquisition = img.acquisition
        if img.path == ref_pathname:
            ref_acquisition = img.acquisition
    if zarr_acquisition is None:
        raise ValueError(f"{zarr_url=} well metadata does not contain expected path and acquisition naming")
    if ref_acquisition is None:
        raise ValueError(f"{ref_url=} well metadata does not contain expected path and acquisition naming")

    return zarr_acquisition, ref_acquisition


