"""Fractal task to convert 3D segmentations into 2D MIP segmentations."""

import logging
from typing import Optional

import ngio
from ngio.utils import NgioFileNotFoundError, ngio_logger
from pydantic import validate_call

ngio_logger.setLevel("ERROR")

logger = logging.getLogger(__name__)


@validate_call
def convert_3d_to_mip(
    zarr_url: str,
    label_name: str,
    level: str = "0",
    new_label_name: Optional[str] = None,
    plate_suffix: str = "_mip",
    image_suffix_3D_to_remove: Optional[str] = None,
    image_suffix_2D_to_add: Optional[str] = None,
    overwrite: bool = False,
) -> None:
    """Convert 3D segmentations into 2D MIP segmentations.

    This task loads the 3D segmentation, projects it along the z-axis, and
    stores it into the 2D MIP OME-Zarr image.

    If the 2D & 3D OME-Zarr images
    have different suffixes in their name, use `image_suffix_3D_to_remove` &
    `image_suffix_2D_to_add`. If their base names are different, this task
    does not support processing them at the moment.

    It makes the assumption that the 3D OME-Zarrs are stored in the same place
    as the 2D OME-Zarrs (same based folder).

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        label_name: Name of the label to copy from 3D OME-Zarr to
            2D OME-Zarr
        new_label_name: Optionally overwriting the name of the label in
            the 2D OME-Zarr
        level: Level of the 3D OME-Zarr label to copy from. Valid choices are
            "0", "1", etc. (depending on which levels are available in the
            OME-Zarr label).
        plate_suffix: Suffix that needs to be added to 3D OME-Zarr name to
            generate the path to the 2D OME-Zarr. If the 2D OME-Zarr is
            "/path/to/my_plate_mip.zarr/B/03/0" and the 3D OME-Zarr is located
            in "/path/to/my_plate.zarr/B/03/0", the correct suffix is "_mip".
        image_suffix_3D_to_remove: If the image name between 2D & 3D don't
            match, this is the suffix that should be removed from the 3D image.
            If the 3D image is in "/path/to/my_plate.zarr/B/03/
            0_registered" and the 2D image is in "/path/to/my_plate_mip.zarr/
            B/03/0", the value should be "_registered"
        image_suffix_2D_to_add: If the image name between 2D & 3D don't
            match, this is the suffix that should be added to the 2D image.
            If the 3D image is in "/path/to/my_plate.zarr/B/03/0" and the
            2D image is in "/path/to/my_plate_mip.zarr/B/03/0_illum_corr", the
            value should be "_illum_corr".
        overwrite: If `True`, overwrite existing label and ROI tables in the
            3D OME-Zarr
    """
    logger.info("Starting 3D to 2D MIP conversion")
    # Normalize zarr_url
    zarr_url = zarr_url.rstrip("/")
    # 0) Preparation
    zarr_2D_url = zarr_url.replace(
        ".zarr",
        f"{plate_suffix}.zarr",
    )  # this is now the 2D zarr (used to be 3D!)
    # Handle changes to image name
    if image_suffix_3D_to_remove:
        zarr_2D_url = zarr_2D_url.rstrip(image_suffix_3D_to_remove)
    if image_suffix_2D_to_add:
        zarr_2D_url += image_suffix_2D_to_add

    if new_label_name is None:
        new_label_name = label_name

    try:
        ome_zarr_container_2d = ngio.open_ome_zarr_container(zarr_2D_url)
    except NgioFileNotFoundError as e:
        raise ValueError(
            f"2D OME-Zarr {zarr_2D_url} not found. Please check the "
            f"suffix (set to {plate_suffix})."
        ) from e

    logger.info(
        f"Copying {label_name} from {zarr_url} to {zarr_2D_url} as "
        f"{new_label_name}."
    )

    # 1) Load a 3D label image
    ome_zarr_container_3d = ngio.open_ome_zarr_container(zarr_url)
    label_img = ome_zarr_container_3d.get_label(label_name, path=level)

    if not label_img.is_3d:
        raise ValueError(
            f"Label image {label_name} is not 3D. It has a shape of "
            f"{label_img.shape} and the axes "
            f"{label_img.axes_mapper.on_disk_axes_names}."
        )

    chunks = list(label_img.chunks)
    label_dask = label_img.get_array(mode="dask")

    # 2) Set up the 2D label image
    ref_image_2d = ome_zarr_container_2d.get_image(
        pixel_size=label_img.pixel_size,
    )

    z_index = label_img.axes_mapper.get_index("z")
    y_index = label_img.axes_mapper.get_index("y")
    x_index = label_img.axes_mapper.get_index("x")
    z_index_2d_reference = ref_image_2d.axes_mapper.get_index("z")
    # since converting to MIP, set chunk size in z to 1
    chunks[z_index] = 1
    chunks = tuple(chunks)

    nb_z_planes = ref_image_2d.shape[z_index_2d_reference]

    shape_2d = (nb_z_planes, label_img.shape[y_index], label_img.shape[x_index])

    pixel_size = label_img.pixel_size
    pixel_size.z = ref_image_2d.pixel_size.z
    axes_names = label_img.axes_mapper.on_disk_axes_names

    new_label_container = ome_zarr_container_2d.derive_label(
        name=new_label_name,
        ref_image=ref_image_2d,
        shape=shape_2d,
        pixel_size=pixel_size,
        axes_names=axes_names,
        chunks=chunks,
        dtype=label_img.dtype,
        overwrite=overwrite,
    )

    # 3) Create the 2D MIP of the label image
    label_img_2D = label_dask.max(axis=z_index, keepdims=True)
    logger.info("Generating maximum intensity projection...")

    # 4) Save changed label image to OME-Zarr
    new_label_container.set_array(label_img_2D, axes_order="zyx")

    logger.info(f"Saved {new_label_name} to 2D Zarr at full resolution")
    # 5) Build pyramids for label image
    new_label_container.consolidate()
    logger.info(f"Built a pyramid for the {new_label_name} label image")

    # 6) Make tables from new label image
    masking_table = ome_zarr_container_2d.build_masking_roi_table(new_label_name)
    new_table_name = f"{new_label_name}_ROI_table"
    ome_zarr_container_2d.add_table(new_table_name, masking_table, overwrite=overwrite)
    logger.info(f"Saved new masking ROI table as {new_table_name}")

    logger.info("Finished 3D to 2D MIP conversion")

    # Give the 2D image as an output so that filters are applied correctly
    # (because manifest type filters get applied to the output image)
    image_list_updates = dict(
        image_list_updates=[
            dict(
                zarr_url=zarr_2D_url,
            )
        ]
    )
    return image_list_updates


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=convert_3d_to_mip,
        logger_name=logger.name,
    )
