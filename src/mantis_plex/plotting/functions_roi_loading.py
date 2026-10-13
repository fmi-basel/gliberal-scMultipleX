from pathlib import Path
from typing import Iterable, List

import anndata as ad
import dask.array as da
import numpy as np
import zarr

### Functions: ROI loading (from Joel's plugin; TODO directly import)

# Functions from Joel's ROI loader Napari plugin
# https://github.com/jluethi/napari-ome-zarr-roi-loader/blob/main/src/napari_ome_zarr_roi_loader


def read_table(zarr_url: Path, roi_table):
    # FIXME: Make this work for cloud-based files => different paths
    table_url = zarr_url / f"tables/{roi_table}"
    return ad.read_zarr(table_url)


def get_metadata(zarr_url):
    with zarr.open(zarr_url) as metadata:
        return metadata


def convert_ROI_table_to_indices(
    ROI: ad.AnnData,
    pxl_sizes_zyx: Iterable[float] = None,
    cols_xyz_pos: Iterable[str] = [
        "x_micrometer",
        "y_micrometer",
        "z_micrometer",
    ],
    cols_xyz_len: Iterable[str] = [
        "len_x_micrometer",
        "len_y_micrometer",
        "len_z_micrometer",
    ],
    reset_origin=False,
) -> List[List[int]]:
    # Function based on
    # https://github.com/fractal-analytics-platform/fractal-tasks-core/blob/main/fractal_tasks_core/lib_regions_of_interest.py
    # Modified to directly use the pixel sizes loaded from the metadata
    # Written by @tcompa (Tommaso Comparin)

    # Set pyramid-level pixel sizes
    pxl_size_z, pxl_size_y, pxl_size_x = pxl_sizes_zyx

    x_pos, y_pos, z_pos = cols_xyz_pos[:]
    x_len, y_len, z_len = cols_xyz_len[:]

    if reset_origin:
        origin_x = min(ROI[:, x_pos].X[:, 0])
        origin_y = min(ROI[:, y_pos].X[:, 0])
        origin_z = min(ROI[:, z_pos].X[:, 0])
    else:
        origin_x = 0.0
        origin_y = 0.0
        origin_z = 0.0

    # list_indices = []
    indices_dict = {}
    for FOV in ROI.obs_names:

        # Extract data from anndata table
        x_micrometer = ROI[FOV, x_pos].X[0, 0] - origin_x
        y_micrometer = ROI[FOV, y_pos].X[0, 0] - origin_y
        z_micrometer = ROI[FOV, z_pos].X[0, 0] - origin_z
        len_x_micrometer = ROI[FOV, x_len].X[0, 0]
        len_y_micrometer = ROI[FOV, y_len].X[0, 0]
        len_z_micrometer = ROI[FOV, z_len].X[0, 0]

        # Identify indices along the three dimensions
        start_x = x_micrometer / pxl_size_x
        end_x = (x_micrometer + len_x_micrometer) / pxl_size_x
        start_y = y_micrometer / pxl_size_y
        end_y = (y_micrometer + len_y_micrometer) / pxl_size_y
        start_z = z_micrometer / pxl_size_z
        end_z = (z_micrometer + len_z_micrometer) / pxl_size_z
        indices = [start_z, end_z, start_y, end_y, start_x, end_x]

        # Round indices to lower integer
        indices = list(map(round, indices))

        # Append ROI indices to to list
        # ist_indices.append(indices[:])
        indices_dict[FOV] = indices[:]

    return indices_dict


def load_intensity_roi(
    zarr_url,
    roi_of_interest,
    channel_index,
    level=0,
    roi_table="FOV_ROI_table",
    reset_origin=False,
):
    # Loads the intensity image of a given ROI in a well
    # returns the image as a numpy array + a list of the image scale

    # image_index defaults to 0 (Change if you have more than one
    # image per well) => FIXME for multiplexing

    # Get the ROI table
    roi_an = read_table(zarr_url, roi_table)

    # Load the pixel sizes from the OME-Zarr file
    dataset = 0  # FIXME, hard coded in case multiple multiscale
    # datasets would be present & multiscales is a list
    metadata = get_metadata(zarr_url)
    scale_img = metadata.attrs["multiscales"][dataset]["datasets"][level][
        "coordinateTransformations"
    ][0]["scale"]

    # FIXME: This is a hack to deal with the fact that the scale can contain
    # the channel as well and out processing functions don't handle that well.
    if len(scale_img) == 4:
        scale_img = scale_img[1:]

    # Get ROI indices for labels
    # FIXME: Switch to a more robust way of loading indices when the
    # dimensionality of the image can vary. This only works for 3D images
    # (all Yokogawa images are saved as 3D images) and
    # by accident for 2D MD images (if they are multichannel)
    # See issue 420 on fractal-tasks-core
    indices_dict = convert_ROI_table_to_indices(
        roi_an,
        pxl_sizes_zyx=scale_img,
        reset_origin=reset_origin,
    )

    # Get the indices for a given roi
    indices = indices_dict[roi_of_interest]
    s_z, e_z, s_y, e_y, s_x, e_x = indices[:]

    # Load data
    img_data_zyx = da.from_zarr(f"{zarr_url}/{level}")[channel_index]
    if len(img_data_zyx.shape) == 2:
        img_roi = img_data_zyx[s_y:e_y, s_x:e_x]
        # FIXME: Hacky way to drop the channel dimension from the scale
        # (for MD data)
        scale_img = scale_img[1:]
    else:
        img_roi = img_data_zyx[s_z:e_z, s_y:e_y, s_x:e_x]

    return np.array(img_roi), scale_img
