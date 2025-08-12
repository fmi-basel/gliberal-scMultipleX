# Copyright 2025 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Nicole Repina <nicole.repina@fmi.ch>
# Tommaso Comparin <tommaso.comparin@exact-lab.it>
# Joel Lüthi <joel.luethi@uzh.ch>
#
import logging
import os as os
import re
import shutil
from functools import reduce
from pathlib import Path
from typing import Sequence, Tuple

import anndata as ad
import dask.array as da
import ngio
import numpy as np
import pandas as pd
import zarr
from fractal_tasks_core.channels import OmeroChannel, get_channel_from_image_zarr
from fractal_tasks_core.labels import prepare_label_group
from fractal_tasks_core.ngff import (
    ZarrGroupNotFoundError,
    load_NgffImageMeta,
    load_NgffWellMeta,
)
from fractal_tasks_core.roi import (
    array_to_bounding_box_table,
    check_valid_ROI_indices,
    convert_indices_to_regions,
    convert_ROI_table_to_indices,
    create_roi_table_from_df_list,
    empty_bounding_box_table,
    load_region,
)
from fractal_tasks_core.tables import write_table
from fractal_tasks_core.tasks._zarr_utils import _update_well_metadata
from fractal_tasks_core.utils import _split_well_path_image_path
from zarr.errors import ArrayNotFoundError

from scmultiplex.meshing.MeshFunctions import (
    downsample_mesh,
    export_stl_polydata,
    export_vtk_polydata,
    labels_to_mesh,
    upsample_mesh,
)

logger = logging.getLogger(__name__)


def read_table_and_attrs(zarr_url, roi_table):
    if isinstance(zarr_url, Path):
        table_url = zarr_url / f"tables/{roi_table}"
    else:
        table_url = f"{zarr_url}/tables/{roi_table}"
    table = ad.read_zarr(table_url)
    table_attrs = get_zattrs(table_url)
    return table, table_attrs


# TODO update relabel_by_linking_consensus task to use these functions
def get_zattrs(zarr_url):
    with zarr.open(zarr_url, mode="r") as zarr_img:
        return zarr_img.attrs.asdict()


def copy_omero_zattrs_from_source_to_target(source_zarr_url, target_zarr_url):
    source_zarr_img = zarr.open(source_zarr_url, mode="r")
    source_zattrs = source_zarr_img.attrs.asdict()
    target_zarr_img = zarr.open(target_zarr_url, mode="r+")
    target_zattrs = target_zarr_img.attrs.asdict()
    target_zattrs["omero"] = source_zattrs["omero"]
    target_zarr_img.attrs.update(target_zattrs)
    return


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
    # Remove empty dataframes
    bbox_dataframe_list = [df for df in bbox_dataframe_list if not df.empty]
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
    zarr_url,
    shape,
    chunks,
    label_dtype,
    inherit_from_label,
    output_label_name,
    logger,
):
    store = zarr.storage.FSStore(f"{zarr_url}/labels/{output_label_name}/0")

    if len(shape) != 3 or len(chunks) != 3:
        raise ValueError(
            "Array must be 3D; if input image is 2D, the 3rd dimension must be set to 1. "
        )

    if shape[0] == 1:
        logger.info("Initializing 2D label image array.")
    else:
        logger.info("Initializing 3D label image array.")

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


def roi_to_pixel_slices(
    roi: ngio.Roi, spacing: np.ndarray
) -> Tuple[slice, slice, slice]:
    """
    Convert Roi (with x,y,z and lengths in um) to pixel slices.

    Args:
        roi: object with attributes roi.x, roi.y, roi.z, roi.x_length, roi.y_length, roi.z_length (all in um)
        spacing: numpy array of shape (3,), with elements [um/pix for z, y, x]

    Returns:
        Tuple of slices: (z_slice, y_slice, x_slice) in pixels
    """
    # Convert start positions (um -> pix)
    start_idx = np.array([roi.z, roi.y, roi.x]) / spacing  # order: z, y, x
    start_idx = np.round(start_idx).astype(int)
    # Convert lengths (um -> pix)
    length_pix = np.array([roi.z_length, roi.y_length, roi.x_length]) / spacing
    length_pix = np.round(length_pix).astype(int)
    # Compute stop position (in pix)
    stop_idx = start_idx + length_pix

    # Build slices
    z_slice = slice(start_idx[0], stop_idx[0])
    y_slice = slice(start_idx[1], stop_idx[1])
    x_slice = slice(start_idx[2], stop_idx[2])

    return (z_slice, y_slice, x_slice)


def save_new_multichannel_image_with_overlap(
    new_npimg: np.ndarray,
    image_url: str,
    region: Tuple[slice, ...],
    apply_to_all_channels: bool = True,
):
    """
    Write multichannel numpy array to zarr in specific ROI region.
    To be used within for loop over ROIs.
    Load on-disk region and combine with new array using element-wise max to handle potential overlapping regions
    that were already written to disk from previous ROIs.
    Apply to all channels
    """
    if apply_to_all_channels:
        assert len(region) == 3  # region should have z,y,x coordinates
        # add empty channel dimension to region; this loads all channels
        region = (slice(None),) + region

    # Load dask from disk, will contain rois of the previously processed objects within for loop
    image_url_level0 = os.path.join(image_url, "0")
    image_array = zarr.open_array(image_url_level0)

    # Load region of current object from disk, will include any previously processed neighboring objects
    region_on_disk = image_array[region]

    # Check that dimensions of rois match
    if region_on_disk.shape != new_npimg.shape:
        raise ValueError(
            f"Shape of new image {new_npimg.shape} must match region {region_on_disk.shape}. "
        )

    # Combine new image with the region that is already on disk using element-wise max
    result = np.fmax(new_npimg, region_on_disk)

    # Store 0-th level of region on disk
    # TODO: Is on-disk chunking preserved here?
    da.array(result).to_zarr(
        url=image_array,
        region=region,
        overwrite=True,
        compute=True,
    )
    return


def save_z_slice_to_label(
    new_zslice: np.ndarray,
    zslice_index: int,
    image_url_level_0: str,
):
    """
    Write zslice to label zarr image.
    """
    # Load label dask from disk
    image_array = zarr.open_array(image_url_level_0)

    z_slice = slice(
        zslice_index, zslice_index + 1
    )  # E.g. first zslice is z=0 to z=1 (non-inclusive) → 1 slice
    y_slice = slice(0, new_zslice.shape[0])  # All y
    x_slice = slice(0, new_zslice.shape[1])  # All x

    region = (z_slice, y_slice, x_slice)

    new_zslice = np.expand_dims(new_zslice, axis=0)  # Add empty z dim

    # Store 0-th level of region on disk
    # TODO: Is on-disk chunking preserved here?
    da.array(new_zslice).to_zarr(
        url=image_array,
        region=region,
        overwrite=True,
        compute=True,
    )
    return


def save_zchunk_to_label(
    new_chunk: np.ndarray | da.Array,
    z_chunk_start_index: int,
    image_url_level_0: str,
):
    """
    Write multichannel numpy array to zarr in specific ROI region.
    To be used within for loop over ROIs.
    Load on-disk region and combine with new array using element-wise max to handle potential overlapping regions
    that were already written to disk from previous ROIs.
    Apply to all channels
    """

    assert len(new_chunk.shape) == 3  # region should have z,y,x coordinates

    # Load label dask from disk
    image_array = zarr.open_array(image_url_level_0)

    z_slice_end_index = z_chunk_start_index + new_chunk.shape[0]

    z_slice = slice(z_chunk_start_index, z_slice_end_index)  # z chunk
    y_slice = slice(0, new_chunk.shape[1])  # All y
    x_slice = slice(0, new_chunk.shape[2])  # All x

    region = (z_slice, y_slice, x_slice)

    # Store 0-th level of region on disk
    da.array(new_chunk).to_zarr(
        url=image_array,
        region=region,
        overwrite=True,
        compute=True,
    )
    return


def compute_and_save_mesh(
    label_image,
    label_str,
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
    resample_mesh_to_target_point_count,
    target_point_count,
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

    number_of_points = mesh_polydata.GetNumberOfPoints()

    logger.info(
        f"Successfully generated surface mesh for object label {label_str} with {number_of_points} points "
        f"and {mesh_polydata.GetNumberOfCells()} triangles."
    )

    if resample_mesh_to_target_point_count:
        # Decide whether to upsample or downsample
        if number_of_points < target_point_count:
            # Upsampling with Loop Subdivision and Quadric Decimation
            mesh_polydata = upsample_mesh(mesh_polydata, target_point_count)
            logger.info(
                f"Upsampled mesh to {mesh_polydata.GetNumberOfPoints()} points "
                f"and {mesh_polydata.GetNumberOfCells()} triangles."
            )
        elif number_of_points > target_point_count:
            # Downsampling with Quadric Decimation
            mesh_polydata = downsample_mesh(mesh_polydata, target_point_count)
            logger.info(
                f"Downsampled mesh to {mesh_polydata.GetNumberOfPoints()} points "
                f"and {mesh_polydata.GetNumberOfCells()} triangles."
            )

        else:
            logger.info("No resampling needed")

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
    try:
        label_dask = da.from_zarr(f"{zarr_url}/labels/{label_name}/{level}")
    except ArrayNotFoundError as e:
        logger.warning(
            "Label not found, exit from the task for this zarr url.\n"
            f"Original error: {str(e)}"
        )
        return {}

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


def read_table_from_zarrurl(zarr_url, table_name):
    path = f"{zarr_url}/tables/{table_name}"
    if os.path.exists(path):
        adata = ad.read_zarr(path)
    else:
        raise ValueError(f"Table {table_name} not found in {zarr_url=}")
    return adata


def check_match_zarr_chunking(dask1, dask2):
    if (dask1.shape != dask2.shape) or (dask1.chunks != dask2.chunks):
        raise ValueError(
            f"Arrays do match: array {dask1} with shape {dask1.shape} and chunk sizes {dask1.chunks}"
            f"does not correspond to array {dask2} with shape {dask2.shape} and chunk size {dask2.chunks}"
        )
    return


def get_channel_index_from_inputmodel(zarr_url, channel_input_model):
    omero_channel = get_channel_from_image_zarr(
        image_zarr_path=zarr_url,
        label=channel_input_model.label,
        wavelength_id=channel_input_model.wavelength_id,
    )
    return omero_channel.index


def replace_origin_by_output(origin_zarr_url, output_zarr_url):
    # Replace origin (input) zarr with the newly created Zarr image
    os.rename(origin_zarr_url, f"{origin_zarr_url}_tmp")
    os.rename(output_zarr_url, origin_zarr_url)
    shutil.rmtree(f"{origin_zarr_url}_tmp")
    return


def update_fractal_well_metadata(zarr_url, output_zarr_url):
    # Keep z-illumination corrected zarr separate from input zarr
    # In this case, update Fractal image list metadata
    image_list_updates = dict(
        image_list_updates=[dict(zarr_url=output_zarr_url, origin=zarr_url)]
    )
    well_url, in_name = _split_well_path_image_path(zarr_url)
    well_url, out_name = _split_well_path_image_path(output_zarr_url)
    try:
        _update_well_metadata(
            well_url=well_url,
            old_image_path=in_name,
            new_image_path=out_name,
        )
    except ZarrGroupNotFoundError:
        logger.debug(f"{zarr_url} is not in an HCS plate. No well metadata got updated")
    except ValueError:
        logger.debug(
            f"Could not update well metadata, likely because "
            f" {output_zarr_url} was already listed there."
        )

    return image_list_updates


def copy_folder_from_zarrurl(origin_zarr_url, output_zarr_url, folder_name, overwrite):
    """
    Temp fix before ngio, literally copy over folder of os.
    Folder_name is string, e.g 'labels', or 'plots'
    """
    origin_folder_path = f"{origin_zarr_url}/{folder_name}"
    output_folder_path = f"{output_zarr_url}/{folder_name}"

    # if labels present in origin, move them
    if os.path.exists(origin_folder_path):
        # make labels directory in destination path
        try:
            shutil.copytree(origin_folder_path, output_folder_path)
            logger.info(f"Copied entire folder '{folder_name}' to output Zarr.")
        # if folder directory already exists in the target zarr, check content before copying
        except FileExistsError:
            if not overwrite:
                logger.warning(
                    f"Folder '{folder_name}' already exists in output and {overwrite=}. Skipping copy."
                )
                return

            logger.info(
                f"Folder '{folder_name}' exists in output. Overwriting contents..."
            )

            # Copy each file/subdirectory individually
            for item in os.listdir(origin_folder_path):
                if item.startswith("."):
                    continue  # Skip hidden files

                origin_item = os.path.join(origin_folder_path, item)

                if not os.path.isdir(origin_item):
                    logger.warning(
                        f"Skipping {folder_name}/{item} as it is not a directory."
                    )
                    continue

                target_item = os.path.join(output_folder_path, item)

                if os.path.exists(target_item):
                    # If already exists, delete it first, then copy from origin
                    shutil.rmtree(target_item)
                    shutil.copytree(origin_item, target_item)
                    logger.info(f"...Replaced {item} with one from origin.")
                else:
                    # If does not exist simply copy the folder
                    shutil.copytree(origin_item, target_item)
                    logger.info(f"...Copied {item} from origin without replacement.")
    else:
        logger.info(
            f"Folder name {folder_name} does not exist in origin zarr, skipping copying"
        )
    return


def save_masking_roi_table_from_df_list(
    bbox_dataframe_list,
    zarr_url,
    output_roi_table_name,
    output_label_name,
    overwrite=True,
):
    """Save new ROI table to zarr, returns the anndata object bbox_table which is saved to disk."""
    bbox_table = create_roi_table_from_df_list(bbox_dataframe_list)
    # Ensure that adata index is string
    bbox_table.obs.index = bbox_table.obs.index.astype(str)
    # Write to zarr group
    image_group = zarr.group(zarr_url)
    logger.info(
        f"Now writing masking bounding-box ROI table with {bbox_table.shape[0]} objects to "
        f"{zarr_url}/tables/{output_roi_table_name}"
    )

    table_attrs = {
        "type": "masking_roi_table",
        "region": {"path": f"../labels/{output_label_name}"},
        "instance_key": "label",
    }

    write_table(
        image_group,
        output_roi_table_name,
        bbox_table,
        overwrite=overwrite,
        table_attrs=table_attrs,
    )
    return bbox_table


def correct_label_column(adata_roi_table, column_name="label"):
    if column_name not in adata_roi_table.obs.columns:
        if adata_roi_table.obs.index.name == column_name:
            # Workaround for new ngio table
            adata_roi_table.obs[column_name] = adata_roi_table.obs.index
        else:
            raise ValueError(
                f"Column name {column_name} "
                f" missing in {adata_roi_table.obs.columns=}"
            )
    return adata_roi_table


def clear_mesh_folder(mesh_folder_name, zarr_url):
    folder_path = f"{zarr_url}/meshes/{mesh_folder_name}"

    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        # Loop through the contents and remove them
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # remove file or symbolic link
            except Exception as e:
                logger.warning(f"Failed to delete {file_path}. Reason: {e}")
    return


# TODO: combine with clear_mesh_folder to make general function
def clear_registration_folder(registration_folder_name, zarr_url):
    folder_path = f"{zarr_url}/registration/{registration_folder_name}"

    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        # Loop through the contents and remove them
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # remove file or symbolic link
            except Exception as e:
                logger.warning(f"Failed to delete {file_path}. Reason: {e}")
    return


# from https://github.com/fractal-analytics-platform/fractal-tasks-core/blob/main/fractal_tasks_core/tasks/_registration_utils.py
def calculate_physical_shifts(
    shifts: np.array,
    level: int,
    coarsening_xy: int,
    full_res_pxl_sizes_zyx: list[float],
) -> list[float]:
    """
    Calculates shifts in physical units based on pixel shifts

    Args:
        shifts: array of shifts, zyx or yx
        level: resolution level
        coarsening_xy: coarsening factor between levels
        full_res_pxl_sizes_zyx: pixel sizes in physical units as zyx

    Returns:
        shifts in physical units as zyx
    """

    curr_pixel_size = np.array(full_res_pxl_sizes_zyx) * coarsening_xy**level
    if len(shifts) == 3:
        shifts_physical = shifts * curr_pixel_size
    elif len(shifts) == 2:
        shifts_physical = [
            0,
            shifts[0] * curr_pixel_size[1],
            shifts[1] * curr_pixel_size[2],
        ]
    else:
        raise ValueError(f"Wrong input for calculate_physical_shifts ({shifts=})")
    return shifts_physical


# from https://github.com/fractal-analytics-platform/fractal-tasks-core/blob/main/fractal_tasks_core/tasks/_registration_utils.py
def get_ROI_table_with_translation(
    ROI_table: ad.AnnData,
    new_shifts: dict[str, list[float]],
) -> ad.AnnData:
    """
    Adds translation columns to a ROI table

    Args:
        ROI_table: Fractal ROI table
        new_shifts: zyx list of shifts

    Returns:
        Fractal ROI table with 3 additional columns for calculated translations
    """

    shift_table = pd.DataFrame(new_shifts).T
    shift_table.columns = ["translation_z", "translation_y", "translation_x"]
    shift_table = shift_table.rename_axis("FieldIndex")
    new_roi_table = ROI_table.to_df().merge(
        shift_table, left_index=True, right_index=True
    )
    if len(new_roi_table) != len(ROI_table):
        raise ValueError(
            "New ROI table with registration info has a "
            f"different length ({len(new_roi_table)=}) "
            f"from the original ROI table ({len(ROI_table)=})"
        )

    adata = ad.AnnData(X=new_roi_table.astype(np.float32))
    adata.obs_names = new_roi_table.index
    adata.var_names = list(map(str, new_roi_table.columns))
    return adata


def remove_roi_table_suffix(name: str) -> str:
    return re.sub(r"_ROI_table", "", name)


def get_num_z_chunks(zarr_array) -> int:
    """
    Return the number of chunks along the Z axis (first dimension)
    for a 3D Zarr array.

    Parameters
    ----------
    zarr_array : zarr.Array or NGIO object
        A Zarr array with shape (Z, Y, X) and chunking enabled.

    Returns
    -------
    num_chunks_z : int
        Number of chunks along the Z (axis=0) dimension.
    """
    z_dim = zarr_array.shape[0]
    z_chunk = zarr_array.chunks[0]

    num_chunks_z = (z_dim + z_chunk - 1) // z_chunk  # ceil division
    return num_chunks_z


def iterate_z_chunks(dask_array: da.Array):
    """
    Iterate over Z chunks of a 3D Dask array, loading all XY pixels
    (i.e., all Y and X chunks) for each Z chunk into memory.

    Parameters
    ----------
    dask_array : dask.array.Array
        A 3D Dask array with shape (Z, Y, X).

    Yields
    ------
    z_start : int
        Starting index of the Z chunk.
    z_stop : int
        Ending index of the Z chunk (exclusive).
    z_block : np.ndarray
        The Z chunk loaded into memory (shape: (z_chunk_size, Y, X)).
    """
    z_chunks = dask_array.chunks[0]  # tuple of chunk sizes along Z
    z_starts = [0] + list(np.cumsum(z_chunks[:-1]))  # starting indices of each Z chunk

    for z_start, z_chunk_size in zip(z_starts, z_chunks):
        z_stop = z_start + z_chunk_size

        # Slice this Z chunk and load entire XY pixels (may span multiple Y,X chunks)
        z_block = dask_array[z_start:z_stop, :, :].compute()

        yield z_start, z_stop, z_block
