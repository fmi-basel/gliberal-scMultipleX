# Copyright (C) 2025 Friedrich Miescher Institute for Biomedical Research

##############################################################################
#                                                                            #
# Author: Nicole Repina              <nicole.repina@fmi.ch>                  #
#                                                                            #
##############################################################################


"""
Calculate z-illumination correction curve for each object in well. Task assumes objects were imaged with a uniform
stain whose intensity drop-off along z-axis is representative of z-correction that needs to be applied.
"""

import datetime
import logging
import os
from typing import Any

import anndata as ad
import pandas as pd
import zarr
from fractal_tasks_core.channels import ChannelInputModel
from fractal_tasks_core.tables import write_table
from fractal_tasks_core.tasks.io_models import InitArgsRegistrationConsensus
from pydantic import validate_call

from scmultiplex.fractal.fractal_helper_functions import (
    format_roi_table,
    load_channel_image,
    load_image_array,
    load_label_rois,
    load_seg_and_raw_region,
)
from scmultiplex.illumination.illum_correction_functions import calculate_correction
from scmultiplex.meshing.LabelFusionFunctions import select_label

logger = logging.getLogger(__name__)


@validate_call
def calculate_z_illumination_correction(
    *,
    # Fractal arguments
    zarr_url: str,
    init_args: InitArgsRegistrationConsensus,
    # Task-specific arguments
    input_channels: list[ChannelInputModel],
    label_name: str = "org",
    roi_table: str = "org_ROI_table",
    percentile: int = 90,
) -> dict[str, Any]:

    """
    Calculate Z-illumination correction task.
    """

    logger.info(
        f"Running for {zarr_url=}. \n"
        f"Calculating z-illumination correction for objects in {label_name=} with {roi_table=}."
    )

    # Always use highest resolution label
    level = 0

    # Always save table with 'zillum' suffix
    output_table_name = "zillum"

    ##############
    # Load segmentation image  ###
    ##############

    (
        label_dask,
        label_adata,
        label_idlist,
        label_ngffmeta,
        label_pixmeta,
    ) = load_label_rois(
        zarr_url,
        label_name,
        roi_table,
        level,
    )

    ##############
    # Load channel image(s)  ###
    ##############

    img_array, ngffmeta_raw, xycoars_raw, pixmeta_raw = load_image_array(
        zarr_url, level
    )

    full_z_count = img_array.shape[-3]

    # Loop over channels
    for channel in input_channels:

        # Load channel dask array and ID list
        ch_dask, ch_idlist = load_channel_image(
            channel,
            img_array,
            zarr_url,
            level,
            roi_table,
            label_adata,
            xycoars_raw,
            pixmeta_raw,
        )

        ##############
        # Iterate over objects and perform segmentation  ###
        ##############

        # Get labels to iterate over
        roi_labels = label_adata.obs_vector("label")
        total_label_count = len(roi_labels)
        compute = True
        object_count = 0

        if channel.label is not None:
            channel_id = channel.label
        elif channel.wavelength_id is not None:
            channel_id = channel.wavelength_id
        else:
            raise ValueError("Channel could not be identified.")

        # make directory for saving z-illumination plots
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(zarr_url, "plots", f"{channel_id}_{timestamp}")

        os.makedirs(filepath, exist_ok=True)

        logger.info(
            f"Starting iteration over {total_label_count} detected objects in ROI table for channel {channel_id}."
        )

        df_correction = []

        # For each object in input ROI table...
        for row in label_adata.obs_names:
            row_int = int(row)
            label_str = roi_labels[row_int]

            seg_numpy, raw_numpy = load_seg_and_raw_region(
                label_dask, ch_dask, label_idlist, ch_idlist, row_int, compute
            )

            if seg_numpy.shape != raw_numpy.shape:
                raise ValueError("Shape of label and raw images must match.")

            # Check that label exists in object
            if float(label_str) not in seg_numpy:
                raise ValueError(
                    f"Object ID {label_str} does not exist in loaded segmentation image. Does input ROI "
                    f"table match label map?"
                )

            seg_numpy = select_label(seg_numpy, label_str)

            masked_image = raw_numpy * seg_numpy

            logger.info(f"Processing object {label_str}.")

            roi_start_z = label_idlist[row_int][0]

            # calculate z-illumination dropoff
            row = calculate_correction(
                masked_image,
                roi_start_z,
                full_z_count,
                label_str,
                filepath,
                percentile=percentile,
            )

            object_count += 1

            # convert to dictionary where key is z, value is i
            df_correction.append(row)

        df_correction = pd.DataFrame(df_correction)

        # follows similar logic as feature extraction task
        if not df_correction.empty:
            correction_table = format_roi_table([df_correction])
        else:
            # Create empty anndata table
            correction_table = ad.AnnData()

        # TODO: check that no values are lower or equal to 0, and nothing greater than 1

        # Write to zarr group
        image_group = zarr.group(f"{zarr_url}")
        table_attrs = {
            "type": "feature_table",
            "fractal_table_version": "1",
            "region": dict(path=f"../labels/{label_name}"),
            "instance_key": "label",
        }

        write_name = f"{output_table_name}_{channel_id}"
        write_table(
            image_group,
            write_name,
            correction_table,
            overwrite=True,
            table_attrs=table_attrs,
        )

        logger.info(
            f"Successfully processed {object_count} out of {total_label_count} labels for channel {channel}."
        )

        logger.info(
            f"Saved z-illumination correction matrix as Anndata table {write_name}"
        )

        logger.info(f"Saved quality control plots per object in {filepath}.")

    logger.info("End calculate_z_illumination_correction task.")

    return {}


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=calculate_z_illumination_correction,
        logger_name=logger.name,
    )
