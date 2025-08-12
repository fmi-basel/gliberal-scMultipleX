# Copyright 2025 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Nicole Repina <nicole.repina@fmi.ch>
#

"""
Copy label image from reference round to moving round and apply rigid transformation (translation and rotation).
"""

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
from skimage.transform import estimate_transform

from scmultiplex.fractal.fractal_helper_functions import (
    iterate_z_chunks,
    save_zchunk_to_label,
)

# Local application imports
from scmultiplex.linking.OrganoidLinkingFunctions import (
    get_euclidean_metrics,
    get_rotation_only_transform,
    get_sorted_label_centroids,
    is_identity_transform,
    resize_array_to_shape,
    select_rotating_euclidean_transform,
    transform_euclidean_metric_to_scipy,
)

# Configure logging
ngio_logger.setLevel("ERROR")
logger = logging.getLogger(__name__)


@validate_call
def shift_by_rigid_shift(
    *,
    # Fractal arguments
    zarr_url: str,
    init_args: InitArgsRegistration,
    # Task-specific arguments
    label_name_to_shift: str,
    new_shifted_label_name: Optional[str] = None,
    label_name_for_2D_rigid_transform: str = None,
    zarr_suffix_to_add: Optional[str] = None,
    image_suffix_to_remove: Optional[str] = None,
    image_suffix_to_add: Optional[str] = None,
    only_apply_rotation: bool = False,
    only_apply_if_rotation_present: bool = False,
):
    """
    Copy label image from reference round to moving round and shift by on the fly rigid transformation.
    Label map from references is thus applied to subsequent moving rounds. 3D label maps can
    be shifted rigid transformation of 2D MIP images, which are loaded from the corresponding 2D zarr
    image. This 2D zarr is found by modifying the zarr and image names, starting from the reference
    zarr url, given the user-defined zarr_suffix_to_add, image_suffix_to_remove, image_suffix_to_add.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        init_args: Intialization arguments provided by
            `_image_based_registration_hcs_init`. They contain the
            reference_zarr_url that is used for registration.
            (standard argument for Fractal tasks, managed by Fractal server).
        label_name_to_shift: Label name of 3D reference round to be copied and shifted.
        new_shifted_label_name: Optionally new name for shifted label.
            If left None, default is {label_name}_shifted
        label_name_for_2D_rigid_transform: Label name for the linked segmentation in the 2D MIP zarr
            to be used for rigid transformation using skimage.transform.EuclideanTransform.
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
        only_apply_rotation: If true, translation output of EuclideanTransform is ignored and only
            rotation is applied, if present.
        only_apply_if_rotation_present: If true, EuclideanTransform is applied (both translation and rotation) to
            if non-zero rotation is detected in the image.
    """

    logger.info(f"Running 'shift_by_rigid_shift' task for {zarr_url=}.")

    # Set OME-Zarr paths
    reference_zarr_url = init_args.reference_zarr_url

    # Open the ome-zarr container
    reference_ome_zarr = open_ome_zarr_container(reference_zarr_url)
    moving_ome_zarr = open_ome_zarr_container(zarr_url)

    # Load label image to copy
    reference_img = reference_ome_zarr.get_label(label_name_to_shift)

    # Load moving image (target)
    moving_img = moving_ome_zarr.get_image()

    # Load 2D label images to calculate rigid shift
    # Get zarr url to label pair (e.g. from _mip zarr), src = reference, dst = current moving round
    src_2d_zarr_url = reference_zarr_url.rstrip("/")
    dst_2d_zarr_url = zarr_url.rstrip("/")
    if zarr_suffix_to_add:
        src_2d_zarr_url = src_2d_zarr_url.replace(
            ".zarr",
            f"{zarr_suffix_to_add}.zarr",
        )  # this is now the path to the registered zarr
        dst_2d_zarr_url = dst_2d_zarr_url.replace(
            ".zarr",
            f"{zarr_suffix_to_add}.zarr",
        )  # this is now the path to the registered zarr
    # Handle changes to image name
    if image_suffix_to_remove:
        src_2d_zarr_url = src_2d_zarr_url.rstrip(image_suffix_to_remove)
        dst_2d_zarr_url = dst_2d_zarr_url.rstrip(image_suffix_to_remove)
    if image_suffix_to_add:
        src_2d_zarr_url += image_suffix_to_add
        dst_2d_zarr_url += image_suffix_to_add

    try:
        ome_zarr_2d_src = ngio.open_ome_zarr_container(src_2d_zarr_url)
    except NgioFileNotFoundError as e:
        raise ValueError(
            f"Registered OME-Zarr {src_2d_zarr_url} not found. Please check the "
            f"suffix (set to {zarr_suffix_to_add=}, {image_suffix_to_remove=}, {image_suffix_to_add=})."
        ) from e

    try:
        ome_zarr_2d_dst = ngio.open_ome_zarr_container(dst_2d_zarr_url)
    except NgioFileNotFoundError as e:
        raise ValueError(
            f"Registered OME-Zarr {dst_2d_zarr_url} not found. Please check the "
            f"suffix (set to {zarr_suffix_to_add=}, {image_suffix_to_remove=}, {image_suffix_to_add=})."
        ) from e

    # Load Labels
    src_label = ome_zarr_2d_src.get_label(name=label_name_for_2D_rigid_transform)
    dst_label = ome_zarr_2d_dst.get_label(name=label_name_for_2D_rigid_transform)
    src_image_np = src_label.get_array(mode="numpy")
    dst_image_np = dst_label.get_array(mode="numpy")

    src_objects = np.unique(src_image_np)
    dst_objects = np.unique(dst_image_np)

    if not np.array_equal(src_objects, dst_objects):
        raise ValueError(
            f"Labels between src and dst 2D arrays are not linked. Detected {len(src_objects)} in "
            f"{src_2d_zarr_url=} and {len(dst_objects)} in {dst_2d_zarr_url=}. "
            f"Rigid transformation with unlinked labels not possible."
        )

    # Get point cloud of organoid centroids
    src = get_sorted_label_centroids(src_image_np)
    dst = get_sorted_label_centroids(dst_image_np)

    deltas = dst - src
    logger.info(
        f"Detected average shift between centroids: {np.mean(deltas, axis=0)} [x,y]"
    )

    # Calculate rigid Euclidean maping from reference (src) to moving (dst) point cloud
    tform = estimate_transform("euclidean", src, dst)
    angle_deg, translation = get_euclidean_metrics(tform)

    logger.info(
        f"Calculated rigid transformation with translation {translation} and "
        f"angle rotation {angle_deg}."
    )

    if only_apply_rotation:
        # Remove translation from tform output
        logger.info("Only applying rotation component of rigid transform.")
        tform = get_rotation_only_transform(tform)

    if only_apply_if_rotation_present:
        logger.info("Determining whether rigid transform has rotation component.")
        tform = select_rotating_euclidean_transform(tform)
        if is_identity_transform(tform):
            logger.info("No rotation component detected.")
        else:
            logger.info(
                "Rotation detected, proceeding with full rigid transformation (translation + rotation)."
            )

    matrix, offset = transform_euclidean_metric_to_scipy(tform)

    # Load reference label array as dask
    label_volume_dask = reference_img.get_array(mode="dask")

    # Save shifted label image in moving round zarr, make new label
    if new_shifted_label_name is None:
        new_shifted_label_name = label_name_to_shift + "_shifted"

    new_label = moving_ome_zarr.derive_label(
        name=new_shifted_label_name, overwrite=True
    )

    # If no rigid transformation necessary, skip writing to disk and just copy over untransformed label image
    if is_identity_transform(tform):

        logger.info(
            "No rigid transformation necessary, resize and directly save to disk"
        )

        # Get shapes of reference and moving label images
        reference_shape = reference_img.shape[-3:]
        moving_shape = moving_img.shape[-3:]

        # Resize array to match shape of moving image
        logger.info(f"Resizing array from {reference_shape=} to {moving_shape=}.")
        label_volume_dask = resize_array_to_shape(label_volume_dask, moving_shape)

        new_label.set_array(label_volume_dask)
    else:
        # Apply rigid transformation
        chunk_size_z = reference_img.chunks[0]
        logger.info(f"Loading xy slices by z chunk size {chunk_size_z}")

        # Get level 0 path of new label to save to
        label_level0_zarr_url = os.path.join(zarr_url, new_label.zarr_array.path)

        # Apply 2D rigid transformation by z slice in each chunk, resize, write each chunk to disk
        logger.info(
            f"Resize each zchunk in XY from reference shape {reference_img.shape[-2:]} to moving shape {moving_img.shape[-2:]}."
        )

        logger.info(
            f"Apply rigid transformation by zslice and by zchunk to reference image {reference_zarr_url}."
        )
        logger.info("Start apply and save to disk...")

        # Loop over z chunks
        for z_start, z_stop, z_data in iterate_z_chunks(label_volume_dask):

            np_chunk_to_save = np.empty_like(
                z_data
            )  # init empty array same size as z chunk
            # Loop over zslices in z chunk
            for z, label_slice in enumerate(z_data):
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

        # for z, label_slice in enumerate(label_volume):
        #     label_slice_transformed = affine_transform(
        #         label_slice,  # source image
        #         matrix=matrix,  # 2x2 rotation
        #         offset=offset,  # translation
        #         order=0,  # nearest-neighbor (good for labels)
        #         mode="constant",  # fill outside with constant value
        #         cval=0,  # the constant value to use (e.g. background label)
        #     )
        #
        #     label_slice_transformed = resize_array_to_shape(label_slice_transformed, moving_img.shape[-2:])
        #
        #     save_z_slice_to_label(new_zslice=label_slice_transformed, zslice_index=z, image_url_level_0=label_level0_zarr_url)

    # Build pyramids for label image
    new_label.consolidate()
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
    wait_seconds = random.randint(1, 15)

    for attempt in range(1, max_retries + 1):
        try:
            reference_ome_zarr.add_table(
                new_table_name, ref_masking_table, overwrite=True
            )
            break  # success, exit loop
        except (FileNotFoundError, OSError, KeyError) as e:
            logging.warning(f"[Attempt {attempt}] Failed to write table due to: {e}")
            if attempt == max_retries:
                raise  # raise error
            time.sleep(wait_seconds)

    logger.info(f"End shift_by_shift task for {zarr_url=}")

    return {}


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=shift_by_rigid_shift,
        logger_name=logger.name,
    )
