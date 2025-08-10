# Copyright (C) 2025 Friedrich Miescher Institute for Biomedical Research

##############################################################################
#                                                                            #
# Author: Nicole Repina              <nicole.repina@fmi.ch>                  #
#                                                                            #
##############################################################################

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
