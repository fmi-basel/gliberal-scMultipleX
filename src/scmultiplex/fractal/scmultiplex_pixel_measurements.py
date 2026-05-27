# Copyright 2026 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Nicole Repina <nicole.repina@fmi.ch>
#

import logging
import warnings
from pathlib import Path
from typing import Dict

import anndata as ad
import fractal_tasks_core
import numpy as np
import pandas as pd
import zarr
from fractal_tasks_core.channels import (
    ChannelNotFoundError,
    get_channel_from_image_zarr,
)
from fractal_tasks_core.tables import write_table
from fractal_tasks_core.upscale_array import upscale_array
from ngio import open_ome_zarr_container
from ngio.utils import ngio_logger
from pydantic import validate_call

from scmultiplex.features.feature_wrapper import get_regionprops_pixel_measurements
from scmultiplex.utils.fractal_utils import FeatureChannelInputModel

__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)
ngio_logger.setLevel("ERROR")


@validate_call
def scmultiplex_pixel_measurements(  # noqa: C901
    *,
    # Default arguments for fractal tasks:
    zarr_url: str,
    # Task-specific arguments:
    label_name: str,
    output_table_name: str,
    input_channels: Dict[str, FeatureChannelInputModel],
    input_roi_table_name: str = "well_ROI_table",
    level: int = 0,
    allow_duplicate_labels: bool = False,
    overwrite: bool = True,
):
    """
    Measure number of pixels above given intensity value for each label object in image. Output is a feature table
    that included the object position (centroid, in metadata units, e.g. um), the number of total pixels in object,
    and the number of pixels above threshold for each input channel. User sets thresholds per channel in task input.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        label_name: Name of the label image to use for measurements.
            Needs to exist in OME-Zarr file
        output_table_name: Name of the output AnnData table to save the
            measurements in.
        input_channels: Dictionary of channels to measure. Keys are the
            names that will be added as prefixes to the measurements,
            values are another dictionary containing either wavelength_id
            or channel_label information to allow Fractal to find the correct
            channel (but not both). Example: {"C01": {"wavelength_id":
            "A01_C01"}. Intensity threshold is used to count positive pixels
            within each object. Pixels with intensity greater than this value are counted as positive.
        input_roi_table_name: Name of the ROI table to loop over. Needs to exist
            as a ROI table in the OME-Zarr file. If it is a masking ROI table, masking of input is performed. Always mask input.
        level: Resolution of the intensity image to load for measurements.
            Only tested for level 0
        allow_duplicate_labels: Set to True to allow saving measurement
            tables with non-unique label values. Can happen when segmentation
            is run on a different ROI than the measurements (e.g. segment
            per well, but measure per FOV)
        overwrite: If `True`, overwrite the task output.
    """
    logger.info(
        f"Running pixel measurement on OME-Zarr image {zarr_url=} at {level=} using {input_roi_table_name=} and segmentation {label_name=}"
    )

    # TODO: consider not always masking input by ROI table and instead make optional (ie. load region but without masking)

    # Load OME Zarr container
    ome_zarr = open_ome_zarr_container(zarr_url)

    # Load ROI tables
    roi_table = ome_zarr.get_table(input_roi_table_name, check_type="generic_roi_table")

    table_type = Path(roi_table._meta.type).name

    # Set masking to True if input is a masking_roi_table
    masking = table_type == "masking_roi_table"

    if masking:
        masking_label_name = Path(roi_table._meta.region.path).name

        channel_image = ome_zarr.get_masked_image(masking_label_name=masking_label_name)
        label_image = ome_zarr.get_masked_label(
            label_name=label_name, masking_label_name=masking_label_name
        )

    else:
        channel_image = ome_zarr.get_image()
        label_image = ome_zarr.get_label(name=label_name)

    # Initialize channel metadata
    # key is user-defined channel name (str), value is channel index (0,1,2..)
    channel_dict = {}
    # key is user-defined channel name (str), value is intensity threshold
    intensity_dict = {}
    logger.info(f"Pixel measurement for channels: {input_channels}")

    for name in input_channels.keys():
        try:
            channel = get_channel_from_image_zarr(
                image_zarr_path=f"{zarr_url}",
                wavelength_id=input_channels[name].wavelength_id,
                label=input_channels[name].label,
            )
        except ChannelNotFoundError as e:
            logger.warning(
                "Channel not found, exit from the task.\n" f"Original error: {str(e)}"
            )
            return {}

        channel_index = channel.index
        channel_dict[name] = channel_index
        intensity_dict[name] = input_channels[name].threshold_intensity

    # Pixel sizes as list: [z,y,x]
    pixel_size = label_image.pixel_size

    # Check whether input image is 2D or 3D
    if label_image.shape[0] == 1:  # if z-dimension is 1, it is actually a 2D image
        is_2d = True
        spacing = (pixel_size.y, pixel_size.x)
    else:
        is_2d = False
        spacing = (pixel_size.z, pixel_size.y, pixel_size.x)

    # Initialize well features
    df_well = pd.DataFrame()
    df_info_well = pd.DataFrame()

    # Loop over regions, e.g. masked roi's, fields of view, or full well overview, as specified by roi table input
    for roi in roi_table.rois():

        # Set inputs
        df_roi = pd.DataFrame()
        df_info_roi = pd.DataFrame()
        first_channel = True

        roi_string = roi.name

        logger.info(f"Processing ROI label {roi_string}...")

        extra_values = {
            "ROI_table_name": input_roi_table_name,
            "ROI_name": roi_string,
        }

        # Load numpy of segmentation images
        if masking:
            label_int = int(roi_string)
            seg = label_image.get_roi_masked(label=label_int)  # z,y,x
        else:
            seg = label_image.get_roi(roi=roi)

        if is_2d:
            seg = np.squeeze(seg, axis=0)  # y,x

        # Feature extraction using seg + channel images
        for c in channel_dict.keys():
            i = channel_dict[c]
            intensity_threshold = intensity_dict[c]

            # Load numpy of channel image
            if masking:
                img = channel_image.get_roi_masked(label=label_int, c=i)  # c=1,z,y,x
            else:
                img = channel_image.get_roi(roi=roi, c=i)

            # Remove empty channel dimension
            img = np.squeeze(img, axis=0)  # z,y,x

            # Remove z axis if 2d image
            if is_2d:
                img = np.squeeze(img, axis=0)

            if seg.shape != img.shape:
                logger.info(
                    f"Upscaling label image from {seg.shape} to match channel image shape of {img.shape}"
                )
                seg = upscale_array(
                    array=seg,
                    target_shape=img.shape,
                    axis=None,
                    pad_with_zeros=True,
                )

            if seg.shape != img.shape:
                raise ValueError(
                    f"Image shape {img.shape} does not match segmentation shape {seg.shape}"
                )

            new_df, new_info_df = get_regionprops_pixel_measurements(
                seg,
                img,
                intensity_threshold=intensity_threshold,
                calculate_area=first_channel,
                is_2D=is_2d,
                spacing=spacing,
                channel_prefix=c,
                extra_values=extra_values,
            )

            # Only measure morphology for the first intensity channel provided
            # => just once per label image
            first_channel = False

            if "label" in df_roi.columns:
                df_roi = df_roi.merge(right=new_df, how="inner")
                df_info_roi = df_info_roi.merge(right=new_info_df, how="inner")
            else:
                df_roi = pd.concat([df_roi, new_df], axis=1)
                df_info_roi = pd.concat([df_info_roi, new_info_df], axis=1)

        df_well = pd.concat([df_well, df_roi], axis=0, ignore_index=True)
        df_info_well = pd.concat([df_info_well, df_info_roi], axis=0, ignore_index=True)

    # TODO: refactor feature table saving to NGIO
    if not df_well.empty and not df_info_well.empty:
        # Ensure that the label column in df_well & df_info_well match
        if not (df_well["label"] == df_info_well["label"]).all():
            raise ValueError(
                "Label column in df_well and df_info_well do not match. "
                f"{df_well['label']} != {df_info_well['label']}"
            )

        # Check that labels are unique
        # Typical issue: Ran segmentation per well, but measurement per FOV
        # => splits labels into multiple measurements
        if not allow_duplicate_labels:
            total_measurements = len(df_well["label"])
            unique_labels = len(df_well["label"].unique())
            if not total_measurements == unique_labels:
                raise ValueError(
                    "Measurement contains non-unique labels: \n"
                    f"{total_measurements =}, {unique_labels =}, "
                )

        df_well.drop(labels=["label"], axis=1, inplace=True)

        # Convert all to float (warning: some would be int, in principle)
        measurement_dtype = np.float32
        df_well = df_well.astype(measurement_dtype)
        df_well.index = df_well.index.map(str)
        # Convert to anndata
        measurement_table = ad.AnnData(df_well)
        measurement_table.obs = df_info_well
    else:
        # Create empty anndata table
        measurement_table = ad.AnnData()

    # Write to zarr group
    image_group = zarr.group(f"{zarr_url}")
    write_table(
        image_group,
        output_table_name,
        measurement_table,
        overwrite=overwrite,
        table_attrs=dict(
            type="feature_table",
            fractal_table_version="1",
            region=dict(path=f"../labels/{label_name}"),
            instance_key="label",
        ),
    )

    logger.info(f"End pixel_measurement task for {zarr_url}/labels/{label_name}")

    return {}


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=scmultiplex_pixel_measurements,
        logger_name=logger.name,
    )
