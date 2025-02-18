# Copyright 2024 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Nicole Repina <nicole.repina@fmi.ch>
# Tommaso Comparin <tommaso.comparin@exact-lab.it>
# Joel Lüthi <joel.luethi@uzh.ch>
#
import logging
import os as os
from functools import reduce
from pathlib import Path
from typing import Sequence

import anndata as ad
import dask.array as da
import numpy as np
import pandas as pd
import zarr
from fractal_tasks_core.channels import OmeroChannel, get_channel_from_image_zarr
from fractal_tasks_core.labels import prepare_label_group
from fractal_tasks_core.ngff import load_NgffImageMeta, load_NgffWellMeta
from fractal_tasks_core.roi import (
    array_to_bounding_box_table,
    check_valid_ROI_indices,
    convert_indices_to_regions,
    convert_ROI_table_to_indices,
    empty_bounding_box_table,
    load_region,
)

from scmultiplex.meshing.MeshFunctions import (
    export_stl_polydata,
    export_vtk_polydata,
    labels_to_mesh,
)

logger = logging.getLogger(__name__)


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
    bbox_table = ad.AnnData(df_well)
    bbox_table.obs = labels

    return bbox_table


def are_linking_table_columns_valid(
    *, table: ad.AnnData, reference_cycle: int, alignment_cycle: int
) -> None:
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
    columns = [
        "R" + str(reference_cycle) + "_label",
        "R" + str(alignment_cycle) + "_label",
    ]
    for column in columns:
        if column not in table.var_names:
            raise ValueError(f"Column {column} is not present in linking table")
    return


def find_consensus(
    *, df_list: Sequence[pd.DataFrame], on: Sequence[str]
) -> pd.DataFrame:
    """
    Find consensus df from a list of dfs where only common ref IDs are kept

    Args:
        df_list: list of dataframes across which consensus is to be found
        on: column name(s) that are in common between rounds
    """

    consensus = reduce(
        lambda left, right: pd.merge(left, right, on=on, how="outer"), df_list
    )

    return consensus


def check_for_duplicates(pd_column):
    """
    Check if input pandas df column contains duplicated entries.
    Returns: is_duplicated, boolean; True if there are 1 or more duplicated item, otherwise False.
    """
    s = pd.Series(pd_column)
    res = s.duplicated()
    is_duplicated = any(res)
    return is_duplicated


def extract_acq_info(zarr_url):
    """
    Find name of acquisition (cycles, e.g. 0, 1, 2, etc) of given paths from their metadata

    Args:
        zarr_url: string, path to zarr image of some acquisition cycle (e.g. /myfolder/test.zarr/C/01/2)

    Output would be 2
    """
    zarr_acquisition = None

    zarr_pathname = Path(zarr_url).name
    wellmeta = load_NgffWellMeta(
        str(Path(zarr_url).parent)
    ).well.images  # list of dictionaries for each round
    for img in wellmeta:
        if img.path == zarr_pathname:
            zarr_acquisition = img.acquisition
    if zarr_acquisition is None:
        raise ValueError(
            f"{zarr_url=} well metadata does not contain expected path and acquisition naming"
        )

    return zarr_acquisition


def initialize_new_label(
    zarr_url, shape, chunks, label_dtype, inherit_from_label, output_label_name, logger
):
    store = zarr.storage.FSStore(f"{zarr_url}/labels/{output_label_name}/0")

    if len(shape) != 3 or len(chunks) != 3 or shape[0] == 1:
        raise ValueError("Expecting 3D image")

    # Add metadata to labels group
    # Get the label_attrs correctly
    # Note that the new label metadata matches the label_name (child) metadata
    label_attrs = get_zattrs(zarr_url=f"{zarr_url}/labels/{inherit_from_label}")
    _ = prepare_label_group(
        image_group=zarr.group(zarr_url),
        label_name=output_label_name,
        overwrite=True,
        label_attrs=label_attrs,
        logger=logger,
    )

    new_label3d_array = zarr.create(
        shape=shape,
        chunks=chunks,
        dtype=label_dtype,
        store=store,
        overwrite=True,
        dimension_separator="/",
    )
    return new_label3d_array


def save_new_label_with_overlap(
    new_npimg,
    new_label3d_array,
    zarr_url,
    output_label_name,
    region,
    compute,
):
    # Load dask from disk, will contain rois of the previously processed objects within for loop
    new_label3d_dask = da.from_zarr(f"{zarr_url}/labels/{output_label_name}/0")
    # Load region of current object from disk, will include any previously processed neighboring objects
    seg_ondisk = load_region(
        data_zyx=new_label3d_dask,
        region=region,
        compute=compute,
    )

    # Check that dimensions of rois match
    if seg_ondisk.shape != new_npimg.shape:
        raise ValueError(
            "Computed label image must match image dimensions of bounding box during saving"
        )

    # Use fmax so that if one of the elements being compared is a NaN, then the non-nan element is returned
    new_npimg_label_tosave = np.fmax(new_npimg, seg_ondisk)

    # Compute and store 0-th level of new 3d label map to disk
    da.array(new_npimg_label_tosave).to_zarr(
        url=new_label3d_array,
        region=region,
        compute=True,
    )
    return


def save_new_label_and_bbox_df(
    new_npimg,
    new_label3d_array,
    zarr_url,
    output_label_name,
    region,
    label_pixmeta,
    compute,
    roi_idlist,
    row_int,
):
    save_new_label_with_overlap(
        new_npimg, new_label3d_array, zarr_url, output_label_name, region, compute
    )

    # make new ROI table
    origin_zyx = convert_indices_to_origin_zyx(roi_idlist[row_int])

    bbox_df = array_to_bounding_box_table(
        new_npimg,
        label_pixmeta,
        origin_zyx=origin_zyx,
    )

    return bbox_df


def compute_and_save_mesh(
    label_image,
    pixmeta,
    polynomial_degree,
    passband,
    feature_angle,
    target_reduction,
    smoothing_iterations,
    zarr_url,
    mesh_folder_name,
    object_name,
    save_as_stl,
):
    # Make mesh with vtkDiscreteFlyingEdges3D algorithm
    # Set spacing to ome-zarr pixel spacing metadata. Mesh will be in physical units (um)
    spacing = tuple(np.array(pixmeta))  # z,y,x e.g. (0.6, 0.216, 0.216)

    # Pad border with 0 so that the mesh forms a manifold
    label_image_padded = np.pad(label_image, 1)

    # Calculate mesh
    mesh_polydata = labels_to_mesh(
        label_image_padded,
        spacing,
        polynomial_degree=polynomial_degree,
        pass_band_param=passband,
        feature_angle=feature_angle,
        target_reduction=target_reduction,
        smoothing_iterations=smoothing_iterations,
        margin=5,
        show_progress=False,
    )
    # Save mesh
    save_transform_path = f"{zarr_url}/meshes/{mesh_folder_name}"
    os.makedirs(save_transform_path, exist_ok=True)

    if save_as_stl is True:
        save_name = f"{int(object_name)}.stl"  # save name is the parent label id
        export_stl_polydata(os.path.join(save_transform_path, save_name), mesh_polydata)
    # Otherwise save as .vtp
    else:
        save_name = f"{int(object_name)}.vtp"  # save name is the parent label id
        export_vtk_polydata(os.path.join(save_transform_path, save_name), mesh_polydata)
    return mesh_polydata


def load_channel_image(
    channel_input_model,
    dask_array,
    zarr_url,
    level,
    roi_table,
    roi_adata,
    xycoars_raw,
    pixmeta_raw,
):

    ##############
    # Load Channel images  ###
    ##############

    # Find channel index for channel
    tmp_channel: OmeroChannel = get_channel_from_image_zarr(
        image_zarr_path=zarr_url,
        wavelength_id=channel_input_model.wavelength_id,
        label=channel_input_model.label,
    )

    channel_id = tmp_channel.index

    # Load channel data
    ch_dask_raw = dask_array[channel_id]

    ch_idlist_raw = convert_ROI_table_to_indices(
        roi_adata,
        level=level,
        coarsening_xy=xycoars_raw,
        full_res_pxl_sizes_zyx=pixmeta_raw,
    )

    check_valid_ROI_indices(ch_idlist_raw, roi_table)

    return ch_dask_raw, ch_idlist_raw


def load_label_rois(
    zarr_url,
    label_name,
    roi_table,
    level,
):

    # Lazily load zarr array for reference cycle
    # load well image as dask array e.g. for nuclear segmentation
    label_dask = da.from_zarr(f"{zarr_url}/labels/{label_name}/{level}")

    # Read ROIs of objects
    roi_adata = ad.read_zarr(f"{zarr_url}/tables/{roi_table}")

    # Read Zarr metadata
    label_ngffmeta = load_NgffImageMeta(f"{zarr_url}/labels/{label_name}")
    label_xycoars = (
        label_ngffmeta.coarsening_xy
    )  # need to know when building new pyramids
    label_pixmeta = label_ngffmeta.get_pixel_sizes_zyx(level=level)

    # Create list of indices for 3D ROIs spanning the entire Z direction
    # Note that this ROI list is generated based on the input ROI table; if the input ROI table is for the group_by
    # objects, then label regions will be loaded based on the group_by ROIs
    roi_idlist = convert_ROI_table_to_indices(
        roi_adata,
        level=level,
        coarsening_xy=label_xycoars,
        full_res_pxl_sizes_zyx=label_pixmeta,
    )

    check_valid_ROI_indices(roi_idlist, roi_table)

    if len(roi_idlist) == 0:
        logger.warning("Well contains no objects")

    return label_dask, roi_adata, roi_idlist, label_ngffmeta, label_pixmeta


def load_image_array(zarr_url, level):
    ngffmeta_raw = load_NgffImageMeta(f"{zarr_url}")
    xycoars_raw = ngffmeta_raw.coarsening_xy
    pixmeta_raw = ngffmeta_raw.get_pixel_sizes_zyx(level=level)

    img_array = da.from_zarr(f"{zarr_url}/{level}")

    return img_array, ngffmeta_raw, xycoars_raw, pixmeta_raw


def load_seg_and_raw_region(
    label_dask, channel_dask, label_idlist, channel_idlist, row_integer, compute
):
    # Load label image of label_name object as numpy array
    seg_numpy = load_region(
        data_zyx=label_dask,
        region=convert_indices_to_regions(label_idlist[row_integer]),
        compute=compute,
    )

    # Load raw image of specific channel for object
    raw_numpy = load_region(
        data_zyx=channel_dask,
        region=convert_indices_to_regions(channel_idlist[row_integer]),
        compute=compute,
    )

    return seg_numpy, raw_numpy
