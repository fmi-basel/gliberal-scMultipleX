"""
Copyright 2023 (C)
    Friedrich Miescher Institute for Biomedical Research

    Original authors:
    Joel Lüthi <joel.luethi@fmi.ch>

    Based on the napari workflow wrapper task, by:
    Tommaso Comparin <tommaso.comparin@exact-lab.it>
    Marco Franzon <marco.franzon@exact-lab.it>

Wrapper of scMultipleX measurements for Fractal
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, Union

import anndata as ad
import fractal_tasks_core
import numpy as np
import pandas as pd
import zarr
from fractal_tasks_core.channels import (
    ChannelInputModel,
    ChannelNotFoundError,
    get_channel_from_image_zarr,
)
from fractal_tasks_core.tables import write_table
from fractal_tasks_core.upscale_array import upscale_array
from ngio import open_ome_zarr_container
from ngio.utils import ngio_logger
from pydantic import validate_call

from scmultiplex.features.feature_wrapper import get_regionprops_measurements

__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)
ngio_logger.setLevel("ERROR")


@validate_call
def scmultiplex_feature_measurements(  # noqa: C901
    *,
    # Default arguments for fractal tasks:
    zarr_url: str,
    # Task-specific arguments:
    label_name: str,
    output_table_name: str,
    input_channels: Union[Dict[str, ChannelInputModel], None] = None,
    input_roi_table_name: str = "well_ROI_table",
    level: int = 0,
    label_level: int = 0,
    measure_morphology: bool = True,
    allow_duplicate_labels: bool = False,
    overwrite: bool = True,
):
    """
    Measurements of intensities and morphologies

    Wrapper task for scmultiplex measurements for Fractal to generate
    measurements of intensities and morphologies

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        label_name: Name of the label image to use for measurements.
            Needs to exist in OME-Zarr file
        output_table_name: Name of the output AnnData table to save the
            measurements in. A table of this name can't exist yet in the
            OME-Zarr file
        input_channels: Dictionary of channels to measure. Keys are the
            names that will be added as prefixes to the measurements,
            values are another dictionary containing either wavelength_id
            or channel_label information to allow Fractal to find the correct
            channel (but not both). Example: {"C01": {"wavelength_id":
            "A01_C01"}. To only measure morphology, provide an empty dict
        input_roi_table_name: Name of the ROI table to loop over. Needs to exist
            as a ROI table in the OME-Zarr file. If it is a masking ROI table, masking of input is performed. Always mask input.
        level: Resolution of the intensity image to load for measurements.
            Only tested for level 0
        label_level: Resolution of the label image to load for measurements.
        measure_morphology: Set to True to measure morphology features
        allow_duplicate_labels: Set to True to allow saving measurement
            tables with non-unique label values. Can happen when segmentation
            is run on a different ROI than the measurements (e.g. segment
            per well, but measure per FOV)
        overwrite: If `True`, overwrite the task output.
    """
    logger.info(f"Running feature extraction on OME-Zarr image {zarr_url=}")

    if input_channels is None and not measure_morphology:
        raise ValueError(
            "You need to either add input_channels to make measurements on "
            "or set measure_morphology to True"
        )

    # Level-related constraint
    logger.info(f"This workflow acts at {level=}")
    if level != 0 or label_level != 0:
        # TODO: Test whether this constraint can be lifted
        logger.warning(
            f"Measuring at {level=} & {label_level=}: It's not recommended "
            "to measure at lower resolutions"
        )

    # TODO: consider not always masking input by ROI table and instead make optional (ie. load region but without masking)

    # TODO throw error if channel is not found!! Or pass

    # Load OME Zarr container
    ome_zarr = open_ome_zarr_container(zarr_url)

    # Load ROI tables
    roi_table = ome_zarr.get_table(input_roi_table_name, check_type="generic_roi_table")

    table_type = Path(roi_table._meta.type).name

    # Set masking to True if input is a masking_roi_table
    masking = table_type == "masking_roi_table"

    if masking:
        masking_label_name = Path(roi_table._meta.region.path).name

        channel_image = ome_zarr.get_masked_image(
            masking_label_name=masking_label_name,
            masking_table_name=input_roi_table_name,
        )
        label_image = ome_zarr.get_masked_label(
            label_name=label_name,
            masking_label_name=masking_label_name,
            masking_table_name=input_roi_table_name,
        )

    else:
        channel_image = ome_zarr.get_image()
        label_image = ome_zarr.get_label(name=label_name)

    if input_channels:
        channel_dict = (
            {}
        )  # key is user-defined channel name (str), value is channel index (0,1,2..)
        logger.info(f"Feature extraction for channels: {input_channels}")

        for name in input_channels.keys():
            try:
                channel = get_channel_from_image_zarr(
                    image_zarr_path=f"{zarr_url}",
                    wavelength_id=input_channels[name].wavelength_id,
                    label=input_channels[name].label,
                )
            except ChannelNotFoundError as e:
                logger.warning(
                    "Channel not found, exit from the task.\n"
                    f"Original error: {str(e)}"
                )
                return {}

            channel_index = channel.index
            channel_dict[name] = channel_index

    else:
        logger.info("Feature extraction only on label map.")

    # Check whether input image is 2D or 3D
    if input_channels:
        # Read spacing meta from channel images
        # Pixel sizes as list: [z,y,x]
        pixel_size = channel_image.pixel_size
    else:
        # Read pixel spacing from label image
        pixel_size = label_image.pixel_size

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
        if input_channels:
            for c in channel_dict.keys():
                i = channel_dict[c]

                # Load numpy of channel image
                if masking:
                    img = channel_image.get_roi_masked(
                        label=label_int, c=i
                    )  # c=1,z,y,x
                else:
                    img = channel_image.get_roi(roi=roi, c=i)

                # Remove empty channel dimension
                img = np.squeeze(img, axis=0)  # z,y,x

                # Remove z axis if 2d image
                if is_2d:
                    img = np.squeeze(img, axis=0)

                calc_morphology = first_channel and measure_morphology

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

                new_df, new_info_df = get_regionprops_measurements(
                    seg,
                    img,
                    spacing=spacing,
                    is_2D=is_2d,
                    measure_morphology=calc_morphology,
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
        else:
            # Only measure morphology
            calc_morphology = first_channel and measure_morphology
            new_df, new_info_df = get_regionprops_measurements(
                seg,
                img=None,
                spacing=spacing,
                is_2D=is_2d,
                measure_morphology=calc_morphology,
                extra_values=extra_values,
            )
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

    logger.info(f"End feature_measurement task for {zarr_url}/labels/{label_name}")

    return {}


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=scmultiplex_feature_measurements,
        logger_name=logger.name,
    )
