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

import json
import logging
from pathlib import Path
from typing import Any, Dict, Sequence

import anndata as ad
import dask.array as da
import fractal_tasks_core
import numpy as np
import pandas as pd
import zarr
from fractal_tasks_core.lib_channels import (
    ChannelNotFoundError,
    get_channel_from_image_zarr,
)
from fractal_tasks_core.lib_input_models import Channel
from fractal_tasks_core.lib_regions_of_interest import (
    convert_ROI_table_to_indices,
    is_ROI_table_valid,
)
from fractal_tasks_core.lib_ROI_overlaps import find_overlaps_in_ROI_indices
from fractal_tasks_core.lib_upscale_array import upscale_array
from fractal_tasks_core.lib_write import write_table
from fractal_tasks_core.lib_zattrs_utils import extract_zyx_pixel_sizes
from pydantic.decorator import validate_arguments

from scmultiplex.features.feature_wrapper import get_regionprops_measurements

__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__


logger = logging.getLogger(__name__)


@validate_arguments
def scmultiplex_feature_measurements(  # noqa: C901
    *,
    # Default arguments for fractal tasks:
    input_paths: Sequence[str],
    output_path: str,
    component: str,
    metadata: Dict[str, Any],
    # Task-specific arguments:
    label_image: str,
    output_table_name: str,
    input_channels: Dict[str, Channel] = None,
    input_ROI_table: str = "well_ROI_table",
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
        input_paths: List of input paths where the image data is stored as
            OME-Zarrs. Should point to the parent folder containing one or many
            OME-Zarr files, not the actual OME-Zarr file. Example:
            `["/some/path/"]`. This task only supports a single input path.
            (standard argument for Fractal tasks, managed by Fractal server).
        output_path: This parameter is not used by this task.
            (standard argument for Fractal tasks, managed by Fractal server).
        component: Path to the OME-Zarr image in the OME-Zarr plate that is
            processed. Example: `"some_plate.zarr/B/03/0"`.
            (standard argument for Fractal tasks, managed by Fractal server).
        metadata: dictionary containing metadata about the OME-Zarr. This task
            requires the following elements to be present in the metadata.
            `coarsening_xy (int)`: coarsening factor in XY of the downsampling
            when building the pyramid. (standard argument for Fractal tasks,
            managed by Fractal server).
        label_image: Name of the label image to use for measurements.
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
        input_ROI_table: Name of the ROI table to loop over. Needs to exists
            as a ROI table in the OME-Zarr file
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
    if input_channels is None and not measure_morphology:
        raise ValueError(
            "You need to either add input_channels to make measurements on "
            "or set measure_morphology to True"
        )

    # 2D intensity image vs. 3D label image
    handle_2D_edge_case = False

    # Level-related constraint
    logger.info(f"This workflow acts at {level=}")
    if level != 0 or label_level != 0:
        # TODO: Test whether this constraint can be lifted
        logger.warning(
            f"Measuring at {level=} & {label_level=}: It's not recommended "
            "to measure at lower resolutions"
        )

    # Pre-processing of task inputs
    if len(input_paths) > 1:
        raise NotImplementedError("We currently only support a single in_path")
    in_path = Path(input_paths[0]).as_posix()
    coarsening_xy = metadata["coarsening_xy"]

    # Check output tables
    group_tables = zarr.group(f"{Path(output_path)}/{component}/tables/")
    current_tables = group_tables.attrs.asdict().get("tables") or []
    if output_table_name in current_tables:
        raise ValueError(
            f"{Path(output_path)}/{component}/tables/ already includes "
            f"{output_table_name=} in {current_tables=}"
        )

    # Load zattrs file and multiscales
    zattrs_file = f"{in_path}/{component}/.zattrs"
    with open(zattrs_file) as jsonfile:
        zattrs = json.load(jsonfile)
    multiscales = zattrs["multiscales"]
    if len(multiscales) > 1:
        raise NotImplementedError(
            f"Found {len(multiscales)} multiscales, "
            "but only one is currently supported."
        )
    if "coordinateTransformations" in multiscales[0].keys():
        raise NotImplementedError(
            "global coordinateTransformations at the multiscales "
            "level are not currently supported"
        )

    # Read ROI table
    roi_table_path = f"{in_path}/{component}/tables/{input_ROI_table}"
    ROI_table = ad.read_zarr(roi_table_path)

    # Read pixel sizes from zattrs file
    if input_channels:
        full_res_pxl_sizes_zyx = extract_zyx_pixel_sizes(zattrs_file, level=0)
    else:
        # The label image may be lower resolution than the intensity images
        zattrs_file_label = f"{in_path}/{component}/labels/{label_image}/.zattrs"
        full_res_pxl_sizes_zyx = extract_zyx_pixel_sizes(zattrs_file_label, level=0)

    # Create list of indices for 3D FOVs spanning the entire Z direction
    list_indices = convert_ROI_table_to_indices(
        ROI_table,
        level=level,
        coarsening_xy=coarsening_xy,
        full_res_pxl_sizes_zyx=full_res_pxl_sizes_zyx,
    )
    num_ROIs = len(list_indices)
    logger.info(
        f"Completed reading ROI table {input_ROI_table}," f" found {num_ROIs} ROIs."
    )
    # Whether to use masking with ROIs
    use_ROI_masks = is_ROI_table_valid(
        table_path=f"{in_path}/{component}/tables/{input_ROI_table}", use_masks=True
    )
    if not use_ROI_masks:
        overlap = find_overlaps_in_ROI_indices(list_indices)
        if overlap:
            raise ValueError(
                f"ROI indices created from {input_ROI_table} table have "
                "overlaps, but we are not using masked loading."
            )

    input_image_arrays = {}
    img_array = da.from_zarr(f"{in_path}/{component}/{level}")
    # Loop over image inputs and assign corresponding channel of the image
    if input_channels:
        for name in input_channels.keys():
            try:
                channel = get_channel_from_image_zarr(
                    image_zarr_path=f"{in_path}/{component}",
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
            input_image_arrays[name] = img_array[channel_index]

            logger.info(f"Prepared input with {name=} and {input_channels[name]=}")
            logger.info(f"{input_image_arrays=}")

    # FIXME: Add check whether label exists?
    input_label_image = da.from_zarr(
        f"{in_path}/{component}/labels/{label_image}/{label_level}"
    )

    if input_channels:
        # Upsample the label image to match the intensity image
        target_shape = list(input_image_arrays.values())[0].shape
        # 2D vs 3D check: Handles the case of 2D images with 3D labels
        if len(target_shape) == 2 and len(input_label_image.shape) == 3:
            handle_2D_edge_case = True
            input_label_image = np.squeeze(input_label_image)
            axis = [0, 1]
        else:
            axis = [1, 2]

        input_label_image = upscale_array(
            array=input_label_image,
            target_shape=target_shape,
            axis=axis,
            pad_with_zeros=True,
        )
        # FIXME: More reliable way to get the correct scale? => switch to ome-zarr-py?
        # Would not work well with multiple different coordinateTransformations
        spacing = multiscales[0]["datasets"][level]["coordinateTransformations"][0][
            "scale"
        ]
        logger.info(f"Loaded {label_image=}")
    else:
        logger.info(
            "No intensity images provided, only calculating measurement for "
            f"the label image {label_image}"
        )
        zattrs_file_label = f"{in_path}/{component}/labels/{label_image}/.zattrs"
        with open(zattrs_file_label) as jsonfile:
            zattrs_label = json.load(jsonfile)
        spacing = zattrs_label["multiscales"][0]["datasets"][level][
            "coordinateTransformations"
        ][0]["scale"]

    # Loop over ROIs to make measurements
    df_well = pd.DataFrame()
    df_info_well = pd.DataFrame()
    for i_ROI, indices in enumerate(list_indices):
        s_z, e_z, s_y, e_y, s_x, e_x = indices[:]
        region = (slice(s_z, e_z), slice(s_y, e_y), slice(s_x, e_x))

        logger.info(f"ROI {i_ROI+1}/{num_ROIs}: {region=}")

        # Define some constant values to be added as a separat column to
        # the obs table
        extra_values = {
            "ROI_table_name": input_ROI_table,
            "ROI_name": ROI_table.obs.index[i_ROI],
        }

        # Load the label image
        if handle_2D_edge_case:
            region = region[1:]

        label_img = input_label_image[region].compute()
        if use_ROI_masks:
            current_label = int(ROI_table.obs.iloc[i_ROI]["label"])
            background = label_img != current_label
            label_img[background] = 0
            extra_values["ROI_label"] = current_label

        if label_img.shape[0] == 1:
            logger.info("Label image is 2D only, processing with 2D options")
            label_img = np.squeeze(label_img, axis=0)
            real_spacing = spacing[-2:]
            is_2D = True
        elif len(label_img.shape) == 2:
            is_2D = True
            real_spacing = spacing[-2:]
        elif len(label_img.shape) == 3:
            is_2D = False
            real_spacing = spacing[-3:]
        else:
            raise NotImplementedError(
                f"Loaded an image of shape {label_img.shape}. "
                "Processing is only supported for 2D & 3D images"
            )

        # Set inputs
        df_roi = pd.DataFrame()
        df_info_roi = pd.DataFrame()
        first_channel = True
        if input_channels:
            for input_name in input_channels.keys():
                img = input_image_arrays[input_name][region].compute()
                # Check whether the input is 2D or 3D & matches the label_img
                if img.shape[0] == 1 and is_2D:
                    img = np.squeeze(img, axis=0)
                elif len(img.shape) == 2 and is_2D:
                    pass
                elif len(img.shape) == 3 and not is_2D:
                    pass
                else:
                    raise NotImplementedError(
                        f"Loaded an image of shape {img.shape}. "
                        f"and label image of shape {label_img.shape}."
                        "Processing is only supported for 2D & 3D images"
                        "and where the label image and the input image have"
                        "the same dimensionality."
                    )

                calc_morphology = first_channel and measure_morphology
                new_df, new_info_df = get_regionprops_measurements(
                    label_img,
                    img,
                    spacing=real_spacing,
                    is_2D=is_2D,
                    measure_morphology=calc_morphology,
                    channel_prefix=input_name,
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
                label_img,
                img=None,
                spacing=real_spacing,
                is_2D=is_2D,
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
        measurement_table = ad.AnnData(df_well, dtype=measurement_dtype)
        measurement_table.obs = df_info_well
    else:
        # Create empty anndata table
        measurement_table = ad.AnnData()

    # Write to zarr group
    image_group = zarr.group(f"{in_path}/{component}")
    write_table(
        image_group,
        output_table_name,
        measurement_table,
        overwrite=overwrite,
        logger=logger,
    )


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=scmultiplex_feature_measurements,
        logger_name=logger.name,
    )
