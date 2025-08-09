# Copyright 2025 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Nicole Repina <nicole.repina@fmi.ch>
#

"""
Copy label image from reference round to moving round and shift by pre-calculated x,y,z translation.
"""

import logging
import time
from typing import Optional

import ngio
import numpy as np
from fractal_tasks_core.tasks.io_models import InitArgsRegistration
from ngio import open_ome_zarr_container
from ngio.utils import NgioFileNotFoundError, ngio_logger
from pydantic import validate_call

# Local application imports
from scmultiplex.linking.OrganoidLinkingFunctions import (
    resize_array_to_shape,
    shift_array_3d_dask,
)

# Configure logging
ngio_logger.setLevel("ERROR")
logger = logging.getLogger(__name__)


@validate_call
def shift_by_shift(
    *,
    # Fractal arguments
    zarr_url: str,
    init_args: InitArgsRegistration,
    # Task-specific arguments
    label_name_to_shift: str,
    new_shifted_label_name: Optional[str] = None,
    translation_table_name: str = "well_ROI_table",
    zarr_suffix_to_add: Optional[str] = None,
    image_suffix_to_remove: Optional[str] = None,
    image_suffix_to_add: Optional[str] = None,
):
    """
    Copy label image from reference round to moving round and shift by pre-calculated x,y,z translation
    loaded from the translation ROI table of a zarr that has been registered across multiplexing rounds.
    This translation ROI table is commonly well_ROI_table containing columns
    "translation_z", "translation_y", "translation_x" in physical units. Label map from references is
    thus applied to subsequent moving rounds, each with its own translation table. 3D label maps can
    be shifted using MIP-based shift calculations by loading the table from the corresponding 2D zarr
    roi table. This 2D zarr is found by modifying the zarr and image names, starting from the reference
    zarr url, given the user-defined zarr_suffix_to_add, image_suffix_to_remove, image_suffix_to_add.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        init_args: Intialization arguments provided by
            `_image_based_registration_hcs_init`. They contain the
            reference_zarr_url that is used for registration.
            (standard argument for Fractal tasks, managed by Fractal server).
        label_name_to_shift: Label name of reference round to be copied and shifted.
        new_shifted_label_name: Optionally new name for shifted label.
            If left None, default is {label_name}_shifted
        translation_table_name: ROI table name that contains the x,y,z translation information
            as columns "translation_z", "translation_y", "translation_x" in physical units. These
            are the shifts that should be applied to the moving image to match it to the reference
            round. If shifts are generated with the scMultiplex calculate object linking task, shifts
            are stored in the 'well_ROI_table'.
        zarr_suffix_to_add: Optional suffix that needs to be added to input OME-Zarr name to
            generate the path to the registered OME-Zarr. If the registered OME-Zarr is
            "/path/to/my_plate_mip.zarr/B/03/0" and the input OME-Zarr is located
            in "/path/to/my_plate.zarr/B/03/0", the correct suffix is "_mip".
        image_suffix_to_remove: If the image name between reference & registered zarrs don't
            match, this is the optional suffix that should be removed from the reference image.
            If the reference image is in "/path/to/my_plate.zarr/B/03/
            0_registered" and the registered image is in "/path/to/my_plate_mip.zarr/
            B/03/0", the value should be "_registered"
        image_suffix_to_add: If the image name between reference & registered zarrs don't
            match, this is the optional suffix that should be added to the reference image.
            If the reference image is in "/path/to/my_plate.zarr/B/03/0" and the
            registered image is in "/path/to/my_plate_mip.zarr/B/03/0_illum_corr", the
            value should be "_illum_corr".
    """

    # Set OME-Zarr paths
    reference_zarr_url = init_args.reference_zarr_url

    # Open the ome-zarr container
    reference_ome_zarr = open_ome_zarr_container(reference_zarr_url)
    moving_ome_zarr = open_ome_zarr_container(zarr_url)

    # Load label image to copy
    reference_img = reference_ome_zarr.get_label(label_name_to_shift)

    # Load moving image (target)
    moving_img = moving_ome_zarr.get_image()

    # Load ROI table that contains shift information
    # Get zarr url to desired table (e.g. from _mip zarr)
    registered_zarr_url = zarr_url.rstrip("/")
    if zarr_suffix_to_add:
        registered_zarr_url = registered_zarr_url.replace(
            ".zarr",
            f"{zarr_suffix_to_add}.zarr",
        )  # this is now the path to the registered zarr
    # Handle changes to image name
    if image_suffix_to_remove:
        registered_zarr_url = registered_zarr_url.rstrip(image_suffix_to_remove)
    if image_suffix_to_add:
        registered_zarr_url += image_suffix_to_add

    try:
        ome_zarr_registered = ngio.open_ome_zarr_container(registered_zarr_url)
    except NgioFileNotFoundError as e:
        raise ValueError(
            f"Registered OME-Zarr {registered_zarr_url} not found. Please check the "
            f"suffix (set to {zarr_suffix_to_add=}, {image_suffix_to_remove=}, {image_suffix_to_add=})."
        ) from e

    # load ROI table, as adata
    # roi_table = ad.read_zarr(f"{registered_zarr_url}/tables/{translation_table_name}")
    roi_table = ome_zarr_registered.get_table(translation_table_name)
    roi_table = roi_table._table_backend.load_as_anndata()

    translation_columns = [
        "translation_z",
        "translation_y",
        "translation_x",
    ]

    if roi_table.var.index.isin(translation_columns).sum() != 3:
        raise ValueError(
            f"Table '{translation_table_name}' in {registered_zarr_url=} does not contain the "
            f"translation columns {translation_columns} necessary to use "
            "this task."
        )

    roi_table = roi_table.to_df()

    # Ensure exactly one row
    if len(roi_table) != 1:
        raise ValueError(
            f"Translation table '{translation_table_name}' must contain exactly one row."
        )

    # Shifts as list: [z,y,x]
    shifts = roi_table.iloc[0][translation_columns].tolist()

    # Pixel sizes as list: [z,y,x]
    pixel_size = reference_img.pixel_size
    spacing = np.array([pixel_size.z, pixel_size.y, pixel_size.x])

    if np.any(spacing == 0):
        logger.warning(
            "Some dimensions in pixel spacing are zero. Replacing with 1 to avoid divide by zero errors."
        )
        # replace zeros with ones
        spacing = np.where(spacing == 0, 1, spacing)

    # Divide shifts by spacing and round to nearest integer
    shifts_pixel_units = np.round(np.array(shifts) / spacing).astype(int)

    # Negate shifts since now we are shifting reference TO moving image
    # (shifts were calculated for the inverse problem)
    shifts_pixel_units = -shifts_pixel_units
    shifts_pixel_units = tuple(shifts_pixel_units)  # convert list to tuple

    # Get shapes of reference and moving label images
    reference_shape = reference_img.shape[-3:]
    moving_shape = moving_img.shape[-3:]

    # Load reference label array as dask
    img_array = reference_img.get_array(mode="dask")

    # Shift array; this preserves the shape of the original reference array
    logger.info(f"Shifting reference array by {shifts_pixel_units} pixel units.")
    shifted_img_array = shift_array_3d_dask(img_array, shifts_pixel_units)

    # Resize array to match shape of moving image
    logger.info(f"Resizing array from {reference_shape=} to {moving_shape=}.")
    shifted_img_array = resize_array_to_shape(shifted_img_array, moving_shape)

    # Save final shifted and resized label image in moving round zarr
    if new_shifted_label_name is None:
        new_shifted_label_name = label_name_to_shift + "_shifted"

    new_label_container = moving_ome_zarr.derive_label(
        name=new_shifted_label_name, overwrite=True
    )
    new_label_container.set_array(shifted_img_array)

    # Build pyramids for label image
    new_label_container.consolidate()
    logger.info(f"Built a pyramid for the {new_shifted_label_name} label image")

    # Make ROI table from new label image
    masking_table = moving_ome_zarr.build_masking_roi_table(new_shifted_label_name)
    new_table_name = f"{new_shifted_label_name}_ROI_table"
    moving_ome_zarr.add_table(new_table_name, masking_table, overwrite=True)
    logger.info(f"Saved new masking ROI table as {new_table_name}")

    # Write a ROI table with same name for the reference image
    # This essentially duplicates the ROI table for the "label_name" image, just renaming it to "_shifted"
    # This way all multiplex rounds have same registered ROI table name
    # Label image name in reference round remains unchanged
    # Redundant because gets written for ref round multiple times, for every ref/mov pair. But no
    # race conditions possible because this table is not read in this task.
    ref_masking_table = reference_ome_zarr.build_masking_roi_table(label_name_to_shift)

    max_retries = 10
    wait_seconds = 2

    for attempt in range(1, max_retries + 1):
        try:
            reference_ome_zarr.add_table(
                new_table_name, ref_masking_table, overwrite=True
            )
            break  # success, exit loop
        except (FileNotFoundError, OSError) as e:
            logging.warning(f"[Attempt {attempt}] Failed to write table due to: {e}")
            if attempt == max_retries:
                raise  # raise error
            time.sleep(wait_seconds)

    logger.info(f"End shift_by_shift task for {zarr_url=}")

    return {}


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=shift_by_shift,
        logger_name=logger.name,
    )
