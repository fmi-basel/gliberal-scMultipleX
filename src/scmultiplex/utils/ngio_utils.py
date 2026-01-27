# Copyright (C) 2025 Friedrich Miescher Institute for Biomedical Research

##############################################################################
#                                                                            #
# Author: Nicole Repina              <nicole.repina@fmi.ch>                  #
#                                                                            #
##############################################################################

import json
import logging
from pathlib import Path

from ngio import open_ome_zarr_plate

logger = logging.getLogger(__name__)


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
