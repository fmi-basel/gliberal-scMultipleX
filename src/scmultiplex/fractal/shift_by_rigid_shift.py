# Copyright 2025 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Nicole Repina <nicole.repina@fmi.ch>
#


import logging
import os
import random
import time
from typing import Optional

import ngio
import numpy as np
from fractal_tasks_core.tasks.io_models import InitArgsRegistration
from ngio import open_ome_zarr_container
from ngio.utils import NgioFileNotFoundError, ngio_logger
from pydantic import validate_call
from scipy.ndimage import affine_transform

from scmultiplex.fractal.fractal_helper_functions import (
    iterate_z_chunks,
    save_zchunk_to_label,
)

# Local application imports
from scmultiplex.linking.OrganoidLinkingFunctions import (
    convert_transform_to_pixels,
    get_euclidean_metrics_from_scipy_affine,
    pad_to_match_maximum_xy_extent,
    resize_array_to_shape,
)
from scmultiplex.utils.ngio_utils import load_sequence_coordinatetransform

# Configure logging
ngio_logger.setLevel("ERROR")
logger = logging.getLogger("shift_by_rigid_shift")


@validate_call
def shift_by_rigid_shift(
    *,
    # Fractal arguments
    zarr_url: str,
    init_args: InitArgsRegistration,
    # Task-specific arguments
    label_name_to_shift: list[str],
    registration_name: str = "rigid_2D",
    load_registration_from_other_container: bool = True,
    zarr_suffix_to_add: Optional[str] = None,
    image_suffix_to_remove: Optional[str] = None,
    image_suffix_to_add: Optional[str] = None,
    new_shifted_label_name: Optional[list[str]] = None,
    overwrite_label: bool = False,
):
    """
    Apply 2D rigid transformation calculated by 'Calculate_2D_rigid_shift' task to labels. Task copies label image from the reference
    round to the moving round and shifts it by the rigid transformation.
    Task works for both 2D and 3D label images. 3D labels are shifted by rigid transformation of each z-slice. For this
    3D case, the transform is loaded from the corresponding 2D zarr image. This 2D zarr is found by modifying the
    zarr and image names, starting from the reference 3D
    zarr url, given the user-defined zarr_suffix_to_add, image_suffix_to_remove, image_suffix_to_add.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        init_args: Intialization arguments provided by
            `_image_based_registration_hcs_init`. They contain the
            reference_zarr_url that is used for registration.
            (standard argument for Fractal tasks, managed by Fractal server).
        label_name_to_shift: List of label names in reference round to be copied and shifted with rigid transform.
        load_registration_from_other_container: If True, load transform .JSON from a different Zarr container. For
            example, set to True if task is run on a 3D Zarr but registration calculation was run on its
            corresponding 2D MIP Zarr (i.e. the 'registered zarr').
        zarr_suffix_to_add: Optional suffix that needs to be added to input OME-Zarr container name to
            generate the path to the registered OME-Zarr container. If the registered URL is
            "/path/to/my_plate_mip.zarr/B/03/0" and the input URL is
            "/path/to/my_plate.zarr/B/03/0", the correct suffix is "_mip".
        image_suffix_to_remove: If the image name between input & registered URLs don't
            match, this is the optional suffix that should be removed from the input image name.
            If the input URL is "/path/to/my_plate.zarr/B/03/0_registered" and the registered image is
            in "/path/to/my_plate_mip.zarr/B/03/0", the value should be "_registered"
        image_suffix_to_add: If the image name between input & registered zarrs don't
            match, this is the optional suffix that should be added to the input image.
            If the input image is in "/path/to/my_plate.zarr/B/03/0" and the
            registered image is in "/path/to/my_plate_mip.zarr/B/03/0_illum_corr", the
            value should be "_illum_corr".
        registration_name: Name of folder that contains the output rigid shift JSON file named 'sequence.json'.
            Created as subfolder of 'registration' folder in moving round by 'Calculate 2D Rigid Shift' task.
        new_shifted_label_name: Optional new name for the shifted label. Follows same order as label_name_to_shift.
            If left None, default is to keep the original {label_name}.
        overwrite_label: If True, overwrites label image in moving OME-Zarr with shifted version, if exists.
    """

    logger.info(f"Running 'shift_by_rigid_shift' task for {zarr_url=}.")

    # Set OME-Zarr paths
    reference_zarr_url = init_args.reference_zarr_url

    # Open the ome-zarr container
    reference_ome_zarr = open_ome_zarr_container(reference_zarr_url)
    moving_ome_zarr = open_ome_zarr_container(zarr_url)

    # Load moving image (target)
    moving_img = moving_ome_zarr.get_image()

    # Check image dimensions
    if moving_ome_zarr.is_2d:
        logger.info("Detected 2D label image.")
    elif moving_ome_zarr.is_3d:
        logger.info("Detected 3D label image.")
    else:
        raise ValueError("Unknown zarr format.")

    # Check label naming
    if new_shifted_label_name is not None and len(new_shifted_label_name) != len(
        label_name_to_shift
    ):
        raise ValueError(
            "Lists new_shifted_label_name must have the same length as label_name_to_shift. Check task "
            "inputs for these arguments!"
        )

    # Get zarr url for moving round that contains rigid transform .json
    if load_registration_from_other_container:
        registration_zarr_url = zarr_url.rstrip("/")
        if zarr_suffix_to_add:
            n = registration_zarr_url.count(".zarr")

            if n != 1:
                raise ValueError(
                    f"Expected exactly one '.zarr' in path, found {n}: "
                    f"{registration_zarr_url}"
                )

            registration_zarr_url = registration_zarr_url.replace(
                ".zarr",
                f"{zarr_suffix_to_add}.zarr",
            )  # this is now the path to the registered zarr
        # Handle changes to image name
        if image_suffix_to_remove:
            if not registration_zarr_url.endswith(image_suffix_to_remove):
                raise ValueError(
                    f"{registration_zarr_url!r} does not end with "
                    f"{image_suffix_to_remove!r}."
                )
            registration_zarr_url = registration_zarr_url.removesuffix(
                image_suffix_to_remove
            )
        if image_suffix_to_add:
            registration_zarr_url += image_suffix_to_add
    else:
        # Load from current zarr
        registration_zarr_url = zarr_url

    # Try loading container to check if exists
    try:
        registration_ome_zarr = ngio.open_ome_zarr_container(registration_zarr_url)
    except NgioFileNotFoundError as e:
        raise ValueError(
            f"Registered OME-Zarr {registration_zarr_url} not found. Please check the "
            f"suffix (set to {zarr_suffix_to_add=}, {image_suffix_to_remove=}, {image_suffix_to_add=})."
        ) from e

    # Get pixel sizes
    current_pixel_size = moving_img.pixel_size
    registration_pixel_size = registration_ome_zarr.get_image().pixel_size

    if not np.isclose(
        current_pixel_size.x, registration_pixel_size.x
    ) or not np.isclose(current_pixel_size.y, registration_pixel_size.y):
        raise ValueError(
            "Pixel sizes do not match between the current image and the image registration was calculated on: "
            f"current=(y={current_pixel_size.y}, x={current_pixel_size.x}), "
            f"registration=(y={registration_pixel_size.y}, x={registration_pixel_size.x})."
        )

    # Load rigid transform .json calculated in 'Calculate 2D Rigid Shift' Task
    json_path = os.path.join(
        registration_zarr_url, "registration", registration_name, "sequence.json"
    )
    matrix_um, offset_um = load_sequence_coordinatetransform(json_path)
    matrix, offset = convert_transform_to_pixels(
        matrix_um, offset_um, current_pixel_size.y, current_pixel_size.x
    )

    angle_deg, translation = get_euclidean_metrics_from_scipy_affine(matrix, offset)

    logger.info(
        f"Loaded rigid transformation with translation {translation} and "
        f"angle rotation {angle_deg}."
    )

    shifted_names = (
        [None] * len(label_name_to_shift)
        if new_shifted_label_name is None
        else new_shifted_label_name
    )

    # Loop over each label that was submitted to task and apply transform
    for label_name, shifted_name in zip(
        label_name_to_shift,
        shifted_names,
        strict=True,
    ):

        logger.info(f"Applying rigid transform to label image {label_name}.")

        # Load label image to copy
        reference_img = reference_ome_zarr.get_label(label_name)
        label_volume_dask = reference_img.get_array(mode="dask")

        if label_volume_dask.ndim != 3:
            raise ValueError(
                f"Expected a 3D label array of shape (Z,Y,X) or (1,Y,X), got shape "
                f"{label_volume_dask.shape}."
            )

        # Make new label for shifted label image in moving round zarr
        if shifted_name is None:
            save_name = label_name
        else:
            save_name = shifted_name

        new_label = moving_ome_zarr.derive_label(
            name=save_name, dtype=label_volume_dask.dtype, overwrite=overwrite_label
        )

        # Apply 2D rigid transformation by z slice in each z-block, resize, write each chunk to disk
        # If image is a 2D array, this code still runs, simply over the single z-slice as image would be (1, Y, X).

        # Note: cannot parallelize this transform with dask over chunks, because transform can shift or rotate
        # image beyond chunk border.
        # Instead, load all xy chunks for a small z-range (specified by z-chunking
        # distance), apply transform, and save. This is valid since transform is only applied XY, not in Z.
        # TODO: specify custom z extent in task input; currently z extent defaults to z-chunking of array.

        # Get level 0 path of new label to save to
        label_level0_zarr_url = os.path.join(zarr_url, new_label.zarr_array.path)

        # Zero-pad reference image in the case that moving image is larger. This ensures that transformed reference
        # image is not cropped when the transform shifts the image beyond image border.

        chunk_size_z = reference_img.chunks[0]
        logger.info(
            f"\t{label_name}: Starting iteration over z-blocks of size "
            f"[z={chunk_size_z}, y={reference_img.shape[-2]}, x={reference_img.shape[-1]}]..."
        )

        label_volume_dask = pad_to_match_maximum_xy_extent(
            label_volume_dask, moving_img.shape[-3:]
        )

        z_loop = 0

        # Loop over z blocks: this loads full z-block of reference label into memory from disk
        for z_start, z_stop, z_data in iterate_z_chunks(label_volume_dask):

            logger.info(f"\t{label_name}: Starting z-block count {z_loop + 1}... ")

            np_chunk_to_save = np.empty_like(
                z_data
            )  # init empty array same size as z-block
            # Loop over zslices in z-block
            for z, label_slice in enumerate(z_data):
                # Apply transform to each z-slice of full well image
                np_chunk_to_save[z] = affine_transform(
                    label_slice,  # source image
                    matrix=matrix,  # 2x2 rotation
                    offset=offset,  # translation
                    order=0,  # nearest-neighbor (good for labels)
                    mode="constant",  # fill outside with constant value
                    cval=0,  # the constant value to use (e.g. background label)
                )

            label_chunk_transformed = resize_array_to_shape(
                np_chunk_to_save,
                (z_data.shape[0], moving_img.shape[-2], moving_img.shape[-1]),
            )  # still numpy
            save_zchunk_to_label(
                new_chunk=label_chunk_transformed,
                z_chunk_start_index=z_start,
                image_url_level_0=label_level0_zarr_url,
            )

            z_loop += 1

        logger.info(
            f"\t{label_name}: Resized xy dimensions of transformed image {reference_img.shape[-2:]} to match destination shape {moving_img.shape[-2:]}."
        )

        # Build pyramids for label image
        new_label.consolidate()
        logger.info(f"\t{label_name}: Built pyramid.")

        # Make ROI table from new label image in moving round
        masking_table = moving_ome_zarr.build_masking_roi_table(save_name)
        new_table_name = f"{save_name}_ROI_table"
        moving_ome_zarr.add_table(new_table_name, masking_table, overwrite=True)
        logger.info(
            f"\t{label_name}: Saved new masking ROI table as {new_table_name} in moving round."
        )

        if shifted_name is not None:
            # In case that a new label name is required, make a
            # new ROI table with same name for the REFERENCE image
            # This essentially duplicates the ROI table for the "label_name" image, just renaming it to "_shifted"
            # This way all multiplex rounds have same registered ROI table name
            # Label image name in reference round remains unchanged
            # Redundant because gets written for ref round multiple times, for every ref/mov pair. But no
            # race conditions possible because this table is not read in this task.
            ref_masking_table = reference_ome_zarr.build_masking_roi_table(label_name)

            max_retries = 10
            wait_seconds = random.randint(1, 15)

            for attempt in range(1, max_retries + 1):
                try:
                    reference_ome_zarr.add_table(
                        new_table_name, ref_masking_table, overwrite=True
                    )
                    break  # success, exit loop
                except (FileNotFoundError, OSError, KeyError) as e:
                    logger.warning(
                        f"[Attempt {attempt}] Failed to write table due to: {e}"
                    )
                    if attempt == max_retries:
                        raise  # raise error
                    time.sleep(wait_seconds)
            logger.info(
                f"\t{label_name}: Saved new masking ROI table as {new_table_name} in reference round."
            )

        logger.info(f"\t{label_name}: Successfully applied rigid transform.")

    logger.info(f"End shift_by_rigid_shift task for {zarr_url=}")

    return {}


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=shift_by_rigid_shift)
