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
from fractal_tasks_core.roi import empty_bounding_box_table


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



