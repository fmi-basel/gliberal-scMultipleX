# Copyright 2026 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Nicole Repina <nicole.repina@fmi.ch>
#

import logging
import os

import numpy as np
from fractal_tasks_core.tasks.io_models import InitArgsRegistration
from ngio import open_ome_zarr_container
from ngio.utils import ngio_logger
from pydantic import validate_call
from skimage.transform import estimate_transform

from scmultiplex.fractal.fractal_helper_functions import clear_registration_folder

# Local application imports
from scmultiplex.linking.OrganoidLinkingFunctions import (
    convert_transform_to_physical,
    get_euclidean_metrics,
    get_sorted_label_centroids,
    transform_tform_to_scipy_affine,
)
from scmultiplex.utils.ngio_utils import save_sequence_coordinatetransform

# Configure logging
ngio_logger.setLevel("ERROR")
logger = logging.getLogger("calculate_2d_rigid_shift")


@validate_call
def calculate_2d_rigid_shift(
    *,
    # Fractal arguments
    zarr_url: str,
    init_args: InitArgsRegistration,
    # Task-specific arguments
    label_name_for_2D_rigid_transform: str,
    registration_name: str = "rigid_2D",
    overwrite_folder: bool = False,
):
    """
    Calculate rigid transformation (xy shift and rotation) between a pair of 2D label images, mapping moving
    image to reference image using skimage.transform.EuclideanTransform.
    Task works on linked label images, so that matching objects have same label ID between rounds.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        init_args: Intialization arguments provided by
            `_image_based_registration_hcs_init`. They contain the
            reference_zarr_url that is used for registration.
            (standard argument for Fractal tasks, managed by Fractal server).
        label_name_for_2D_rigid_transform: Label name for the linked segmentation in the 2D MIP zarr
            to be used for rigid transformation using skimage.transform.EuclideanTransform.
        registration_name: Name of folder that contains the output rigid shift JSON file named 'sequence.json'.
            Created as subfolder of 'registration' folder in moving round.
        overwrite_folder: If True, clear existing subfolder {registration_name} in 'registration' folder to allow
            re-run of registration with same name.
    """

    logger.info(f"Running 'calculate_2d_rigid_shift' task for {zarr_url=}.")

    # Set OME-Zarr paths
    reference_zarr_url = init_args.reference_zarr_url

    # Open the ome-zarr container
    ome_zarr_2d_src = open_ome_zarr_container(reference_zarr_url)
    ome_zarr_2d_dst = open_ome_zarr_container(zarr_url)

    # Load Labels
    src_label = ome_zarr_2d_src.get_label(name=label_name_for_2D_rigid_transform)
    dst_label = ome_zarr_2d_dst.get_label(name=label_name_for_2D_rigid_transform)
    src_image_np = src_label.get_array(mode="numpy")
    dst_image_np = dst_label.get_array(mode="numpy")

    # Ensure arrays are (1, Y, X), not 3D volumes
    for name, arr in [("source", src_image_np), ("destination", dst_image_np)]:
        if arr.ndim != 3:
            raise ValueError(
                f"Expected {name} label image to have 3 dimensions, got {arr.shape}."
            )
        if arr.shape[0] != 1:
            raise ValueError(
                f"Expected {name} label image to have a singleton Z-dim (1, Y, X), "
                f"got {arr.shape}. 3D label images are not supported."
            )

    # Check pixel sizes
    src_pixel_size = src_label.pixel_size
    dst_pixel_size = dst_label.pixel_size

    if not np.isclose(src_pixel_size.x, dst_pixel_size.x) or not np.isclose(
        src_pixel_size.y, dst_pixel_size.y
    ):
        raise ValueError(
            "Reference and moving label pixel sizes do not match: "
            f"reference=(y={src_pixel_size.y}, x={src_pixel_size.x}), "
            f"moving=(y={dst_pixel_size.y}, x={dst_pixel_size.x})."
        )

    # Get label images
    src_image_np = np.squeeze(src_image_np, axis=0)
    dst_image_np = np.squeeze(dst_image_np, axis=0)

    src_objects = np.unique(src_image_np)
    src_objects = src_objects[src_objects != 0]  # remove background label

    dst_objects = np.unique(dst_image_np)
    dst_objects = dst_objects[dst_objects != 0]  # remove background label

    missing_in_dst = np.setdiff1d(src_objects, dst_objects)
    missing_in_src = np.setdiff1d(dst_objects, src_objects)

    if missing_in_dst.size or missing_in_src.size:
        raise ValueError(
            "Labels between source and destination are not linked.\n"
            f"Missing in destination: {missing_in_dst.tolist()}\n"
            f"Missing in source: {missing_in_src.tolist()}"
        )

    # Get point cloud of organoid centroids
    src = get_sorted_label_centroids(src_image_np)
    dst = get_sorted_label_centroids(dst_image_np)

    deltas = dst - src
    logger.info(
        f"Quality control: Average shift between point cloud centroids = {np.mean(deltas, axis=0)} [x,y]"
    )

    # Calculate rigid Euclidean mapping from reference (src) to moving (dst) point cloud
    logger.info("Computing 2D EuclideanTransform (rigid transformation)...")
    tform = estimate_transform("euclidean", src, dst)

    if not np.all(np.isfinite(tform.params)):
        raise ValueError(
            "Estimated Euclidean transform contains NaN or infinite values."
        )

    angle_deg, translation = get_euclidean_metrics(tform)

    logger.info(
        f"Calculated rigid transformation with translation {translation} and "
        f"angle rotation {angle_deg}."
    )

    # Convert the forward skimage transform (reference -> moving)
    # to the inverse mapping required by scipy.ndimage.affine_transform:
    #
    # [y_reference, x_reference] =
    #     matrix @ [y_moving, x_moving] + offset
    #
    # The saved matrix and offset therefore map moving output coordinates
    # back to reference input coordinates.
    matrix, offset = transform_tform_to_scipy_affine(tform)

    # save transformation to disk, in "registration" folder of 2D dst image
    # convert transform from pixels to physical units using zarr metadata
    matrix_um, offset_um = convert_transform_to_physical(
        matrix, offset, src_pixel_size.y, src_pixel_size.x
    )

    # Set save location and clear if necessary
    registration_save_path = os.path.join(zarr_url, "registration", registration_name)
    if os.path.isdir(registration_save_path):
        if overwrite_folder:
            clear_registration_folder(registration_name, zarr_url)
            logger.info(f"Cleared existing registration folder: {registration_name=}")
        else:
            raise ValueError(
                f"Folder {registration_name=} already exists. To overwrite, set "
                f"overwrite_folder=True."
            )
    else:
        # Make directory
        os.makedirs(registration_save_path)

    json_path = save_sequence_coordinatetransform(
        matrix_um, offset_um, registration_save_path
    )
    logger.info(f"Saving rigid shift as sequence coordinateTransform to: {json_path=}")

    logger.info(f"End calculate_2d_rigid_shift task for {zarr_url=}")

    return {}


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=calculate_2d_rigid_shift)
