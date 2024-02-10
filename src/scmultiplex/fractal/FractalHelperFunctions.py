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

def read_table_and_attrs(zarr_url: Path, roi_table):
    table_url = zarr_url / f"tables/{roi_table}"
    table = ad.read_zarr(table_url)
    table_attrs = get_zattrs(table_url)
    return table, table_attrs

# TODO update relabel_by_linking_consensus task to use these functions
def get_zattrs(zarr_url):
    with zarr.open(zarr_url, mode="r") as zarr_img:
        return zarr_img.attrs.asdict()

