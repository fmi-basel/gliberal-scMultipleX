# Copyright (C) 2025 Friedrich Miescher Institute for Biomedical Research

##############################################################################
#                                                                            #
# Author: Nicole Repina              <nicole.repina@fmi.ch>                  #
#                                                                            #
##############################################################################

import logging
from typing import Optional

import ngio
import numpy as np
from fractal_tasks_core.pyramids import build_pyramid
from ngio.utils import NgioFileNotFoundError, ngio_logger
from pydantic import validate_call

from scmultiplex.fractal.fractal_helper_functions import load_image_array

ngio_logger.setLevel("ERROR")

logger = logging.getLogger(__name__)


@validate_call
def build_label_image(
    # Fractal arguments
    zarr_url: str,
    # Task-specific arguments
    label_name: str,
    build_zarr_pyramid: bool = True,
    compute_masking_roi_table: bool = True,
    new_table_name: Optional[str] = None,
) -> None:
    """
    If label already exists on disk at high resolution, this task can be used to build the pyramid structure and
    compute the corresponding masking ROI table.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
        label_name: Label name of segmentation to be processed.
        build_zarr_pyramid: If True, build pyramid for input label_name.
        compute_masking_roi_table: If True, compute masking ROI table and save.
        new_table_name: Optional name for ROI table. If left None, default is {label_name}_ROI_table.
    """
    # Always use highest resolution label
    level = 0

    img_array, ngffmeta_raw, xycoars_raw, pixmeta_raw = load_image_array(
        zarr_url, level
    )

    label_url = f"{zarr_url}/labels/{label_name}"

    ##############
    # Build pyramid and clean up zarr metadata ###
    ##############
    if build_zarr_pyramid:

        build_pyramid(
            zarrurl=label_url,
            overwrite=True,
            num_levels=ngffmeta_raw.num_levels,
            coarsening_xy=xycoars_raw,
            chunksize=img_array.chunksize,
            aggregation_function=np.max,
        )

    logger.info(f"Built a pyramid for the {label_name} label image")

    ##############
    # Make masking ROI table derived from label image - use ngio ###
    ##############
    if compute_masking_roi_table:

        zarr_url = zarr_url.rstrip("/")

        try:
            ome_zarr_container = ngio.open_ome_zarr_container(zarr_url)
        except NgioFileNotFoundError as e:
            raise ValueError(f"OME-Zarr {zarr_url} not found.") from e

        masking_table = ome_zarr_container.build_masking_roi_table(label_name)

        if new_table_name is None:
            new_table_name = f"{label_name}_ROI_table"

        ome_zarr_container.add_table(new_table_name, masking_table, overwrite=True)

        logger.info(f"Saved new masking ROI table as {new_table_name}")

    return


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=build_label_image,
        logger_name=logger.name,
    )
