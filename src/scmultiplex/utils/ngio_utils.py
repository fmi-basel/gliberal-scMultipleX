# Copyright (C) 2025 Friedrich Miescher Institute for Biomedical Research

##############################################################################
#                                                                            #
# Author: Nicole Repina              <nicole.repina@fmi.ch>                  #
#                                                                            #
##############################################################################

import json
import logging
from pathlib import Path
from typing import List, Tuple, Union

import dask.array as da
import numpy as np
from ngio import open_ome_zarr_plate

logger = logging.getLogger(__name__)

ArrayLike = Union[np.ndarray, da.Array]


def update_well_zattrs_with_new_image(
    zarr_url: str,
    new_image_name: str,
    acquisition_id: int,
):
    """
    Update well .zatts metadata when making a new image in the well. For example when deriving an image
    during illumination correction or registration, e.g. making a new image called "1_registered" derived
    from round 1 image.

    Args:
        zarr_url (str): string path to the original image e.g. ".../myzarr.zarr/A/01/1"
        new_image_name (str): name of the new image e.g. "1_registered"
        acquisition_id (int): acquisition id of the new image e.g. 1
    Returns:
        None
        Updates on-disk metadata to include entry: "acquisition": 1, "path": "1_registered"
    """
    p = Path(zarr_url)
    plate_url = p.parents[2]  # this is path to .zarr
    row = p.parents[1].name  # e.g 'A'
    column = p.parents[0].name  # e.g '01'

    plate_ome_zarr = open_ome_zarr_plate(plate_url)
    # Update well metadata (.zattrs) to add new image, only if does not already exist
    well = plate_ome_zarr.get_well(row=row, column=column)
    if new_image_name not in well.paths():
        plate_ome_zarr.add_image(
            row=row,
            column=column,
            image_path=new_image_name,
            acquisition_id=acquisition_id,
        )
    else:
        logger.info(
            f"Image with name {new_image_name} already exists in well metadata."
        )


def save_sequence_coordinatetransform(matrix_um, offset_um, folder_path):
    """
    Save a forward transform as a JSON file named 'sequence.json' in the specified folder.

    Parameters
    ----------
    matrix_um : np.ndarray
        2x2 rotation matrix in micrometer units
    offset_um : np.ndarray
        2-element translation vector in micrometer units [y, x]
    folder_path : str
        Folder path where 'sequence.json' will be saved
    """
    # Ensure values are plain Python types
    rotation_list = matrix_um.tolist()
    translation_list = offset_um.tolist()

    # Construct JSON dictionary
    transform_json = {
        "coordinateSystems": [
            {"name": "moving", "axes": [{"name": "y"}, {"name": "x"}]},
            {
                "name": "moving_nonrigid",
                "axes": [{"name": "y_nonrigid"}, {"name": "x_nonrigid"}],
            },
        ],
        "coordinateTransformations": [
            {
                "type": "sequence",
                "input": "moving",
                "output": "moving_nonrigid",
                "transformations": [
                    {"type": "rotation", "rotation": rotation_list},
                    {"type": "translation", "translation": translation_list},
                ],
            }
        ],
    }

    # Build full file path
    file_path = f"{folder_path}/sequence.json"

    # Write to file
    with open(file_path, "w") as f:
        json.dump(transform_json, f, indent=4)

    return file_path


def squeeze_with_record(array: ArrayLike) -> Tuple[ArrayLike, List[int]]:
    """
    Remove all singleton dimensions from a Dask array while recording
    which axes were removed so they can be restored later.

    This function behaves like `dask_array.squeeze()`, but additionally
    returns the indices of the axes that were squeezed out.

    Parameters
    ----------
    array : numpy.ndarray or dask.array.Array
        Input array.

    Returns
    -------
    squeezed_array : dask.array.Array
        The squeezed Dask array with all dimensions of size 1 removed.

    squeezed_axes : list of int
        A list of axis indices that were removed. This list can be passed
        to `restore_squeezed_axes` to reconstruct the original shape.

    Notes
    -----
    - This function does not load data into memory.
    - If no singleton dimensions exist, the input array is returned
      unchanged and `squeezed_axes` will be an empty list.
    - Axis numbering refers to the original array shape before squeezing.

    Examples
    --------
    >> import dask.array as da
    >> x = da.zeros((1, 128, 128, 1))
    >> y, axes = squeeze_with_record(x)
    >> y.shape
    (128, 128)
    >> axes
    [0, 3]
    """
    axes = [i for i, size in enumerate(array.shape) if size == 1]
    squeezed_array = array.squeeze()
    return squeezed_array, axes


def restore_squeezed_axes(
    array: ArrayLike,
    axes: List[int],
) -> ArrayLike:
    """
    Restore singleton dimensions that were previously removed from
    a Dask array using `squeeze_with_record`.

    This function reinserts size-1 axes at the positions specified
    in `axes`, reconstructing the original dimensionality.

    Parameters
    ----------
    array : numpy.ndarray or dask.array.Array
        Input array to expand.

    axes : list of int
        List of axis indices that should be reinserted as singleton
        dimensions. Typically obtained from `squeeze_with_record`.

    Returns
    -------
    restored_array : numpy.ndarray or dask.array.Array
        Array with singleton dimensions restored.

    Notes
    -----
    - The function does not allocate memory or compute values.
    - Axes are restored in sorted order to ensure correct placement.
    - If `axes` is empty, the input array is returned unchanged.

    Examples
    --------
    >> x = da.zeros((1, 128, 128, 1))
    >> y, axes = squeeze_with_record(x)
    >> z = restore_squeezed_axes(y, axes)
    >> z.shape
    (1, 128, 128, 1)
    """
    for axis in sorted(axes):
        array = np.expand_dims(array, axis=axis)
    return array
