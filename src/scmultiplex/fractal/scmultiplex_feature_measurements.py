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
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence

import anndata as ad
import dask.array as da
import numpy as np
import pandas as pd
import zarr
from anndata.experimental import write_elem
from scmultiplex.features.feature_wrapper import get_regionprops_measurements

import fractal_tasks_core
from fractal_tasks_core.lib_channels import get_channel_from_image_zarr
from fractal_tasks_core.lib_regions_of_interest import (
    convert_ROI_table_to_indices,
)
from fractal_tasks_core.lib_upscale_array import upscale_array
from fractal_tasks_core.lib_zattrs_utils import extract_zyx_pixel_sizes


__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__


logger = logging.getLogger(__name__)


def scmultiplex_measurements(
    *,
    # Default arguments for fractal tasks:
    input_paths: Sequence[str],
    output_path: str,
    component: str,
    metadata: Dict[str, Any],
    # Task-specific arguments:
    input_ROI_table: str = "FOV_ROI_table",
    input_channels: Dict[str, Dict[str, str]],
    label_image: str,
    output_table_name: str,
    level=0,
    measure_morphology: bool = True,
    allow_duplicate_labels: bool = False,
):
    """
    Wrapper task for scmultiplex measurements for Fractal

    :param input_paths: TBD (default arg for Fractal tasks)
    :param output_path: TBD (default arg for Fractal tasks)
    :param metadata: TBD (default arg for Fractal tasks)
    :param component: TBD (default arg for Fractal tasks)
    :param allow_duplicate_labels: Set to True to allow saving measurement
                                   tables with non-unique label values. Can
                                   happen when segmentation is run on a
                                   different ROI than the measurements
                                   (e.g. segment per well, but measure per FOV)
    """

    # Level-related constraint
    logger.info(f"This workflow acts at {level=}")
    if level != 0:
        # TODO: Test whether this constraint can be lifted
        raise NotImplementedError(
            "scMultipleX Measurements are only implemented for level 0"
        )

    # Pre-processing of task inputs
    if len(input_paths) > 1:
        raise NotImplementedError("We currently only support a single in_path")
    in_path = Path(input_paths[0]).as_posix()
    # num_levels = metadata["num_levels"]
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
    with open(zattrs_file, "r") as jsonfile:
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

    # FIXME: More reliable way to get the correct scale?
    # Would not work well with multiple different coordinateTransformations
    spacing = multiscales[0]["datasets"][level]["coordinateTransformations"][0]["scale"]

    # Read ROI table
    ROI_table = ad.read_zarr(f"{in_path}/{component}/tables/{input_ROI_table}")

    # Read pixel sizes from zattrs file
    full_res_pxl_sizes_zyx = extract_zyx_pixel_sizes(zattrs_file, level=0)

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

    input_image_arrays = {}
    img_array = da.from_zarr(f"{in_path}/{component}/{level}")
    # Loop over image inputs and assign corresponding channel of the image
    for name in input_channels.keys():
        params = input_channels[name]
        if "wavelength_id" in params and "channel_label" in params:
            raise ValueError(
                "One and only one among channel_label and wavelength_id"
                f" attributes must be provided, but input {name} in "
                f"input_channels has {params=}."
            )
        channel = get_channel_from_image_zarr(
            image_zarr_path=f"{in_path}/{component}",
            wavelength_id=params.get("wavelength_id", None),
            label=params.get("channel_label", None),
        )
        channel_index = channel["index"]
        input_image_arrays[name] = img_array[channel_index]

        logger.info(f"Prepared input with {name=} and {params=}")
        logger.info(f"{input_image_arrays=}")

    # Set target_shape for upscaling labels
    if not input_image_arrays:
        # TODO: Allow user to just make morphology measurements?
        raise ValueError(f"No images loaded for {input_channels=}")

    # FIXME: Add check whether label exists?
    input_label_image = da.from_zarr(
        f"{in_path}/{component}/labels/{label_image}/{level}"
    )

    # If we allow shape measurements without intensity images,
    # then need to upscale differently
    target_shape = list(input_image_arrays.values())[0].shape
    input_label_image = upscale_array(
        array=input_label_image,
        target_shape=target_shape,
        axis=[1, 2],
        pad_with_zeros=True,
    )

    logger.info(f"Loaded {label_image=} and {params=}")

    #####

    df_well = pd.DataFrame()
    df_info_well = pd.DataFrame()
    for i_ROI, indices in enumerate(list_indices):
        s_z, e_z, s_y, e_y, s_x, e_x = indices[:]
        region = (slice(s_z, e_z), slice(s_y, e_y), slice(s_x, e_x))

        logger.info(f"ROI {i_ROI+1}/{num_ROIs}: {region=}")

        # Load the label image
        label_img = input_label_image[region].compute()
        if label_img.shape[0] == 1:
            logger.info("Label image is 2D only, processing with 2D options")
            label_img = np.squeeze(label_img, axis=0)

        # Set inputs
        df_roi = pd.DataFrame()
        df_info_roi = pd.DataFrame()
        first_channel = True
        for input_name in input_channels.keys():
            img = input_image_arrays[input_name][region].compute()
            # TODO: Add option to mask with a ROI label mask. Actually only
            # needs input masking here, as the return value is a table

            # Check whether the input is 2D or 3D
            if img.shape[0] == 1:
                logger.info("Image is 2D only, processing with 2D options")
                img = np.squeeze(img, axis=0)
                real_spacing = spacing[1:]
                is_2D = True
            elif len(img.shape) == 2:
                real_spacing = spacing
                is_2D = True
            elif len(img.shape) == 3:
                real_spacing = spacing
                is_2D = False
            else:
                raise NotImplementedError(
                    f"Loaded an image of shape {img.shape}. "
                    "Processing is only supported for 2D & 3D images"
                )

            # Define some constant values to be added as a separat column to
            # the obs table
            # TODO: Add ROI label once we allow for masked ROIs
            extra_values = {
                "ROI_table_name": input_ROI_table,
                "ROI_name": ROI_table.obs.index[i_ROI],
            }

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
                df_roi = df_roi.merge(right=new_df, how="outer", on="label")
                df_info_roi = df_info_roi.merge(
                    right=new_info_df, how="outer", on="label"
                )
            else:
                df_roi = pd.concat([df_roi, new_df], axis=1)
                df_info_roi = pd.concat([df_info_roi, new_info_df], axis=1)

        df_well = pd.concat([df_well, df_roi], axis=0, ignore_index=True)
        df_info_well = pd.concat([df_info_well, df_info_roi], axis=0, ignore_index=True)

    print(df_well)
    print(list(df_well.columns))

    # Ensure that the label column in df_well & df_info_well match
    if not (df_well["label"] == df_info_well["label"]).all():
        raise ValueError(
            f"Label column in df_well and df_info_well do not match. {df_well['label']} != {df_info_well['label']}"
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
    # Convert to anndata
    measurement_table = ad.AnnData(df_well, dtype=measurement_dtype)
    # measurement_table.obs = labels
    measurement_table.obs = df_info_well

    # Write to zarr group
    write_elem(group_tables, output_table_name, measurement_table)
    # Update OME-NGFF metadata
    new_tables = current_tables + [output_table_name]
    group_tables.attrs["tables"] = new_tables


if __name__ == "__main__":
    from pydantic import BaseModel
    from pydantic import Extra
    from fractal_tasks_core._utils import run_fractal_task

    class TaskArguments(BaseModel, extra=Extra.forbid):
        input_paths: Sequence[str]
        output_path: str
        metadata: Dict[str, Any]
        component: str
        input_ROI_table: Optional[str]
        input_channels: Dict[str, Dict[str, str]]
        label_image: str
        output_table_name: str
        level: int
        measure_morphology: bool

    run_fractal_task(
        task_function=scmultiplex_measurements,
        TaskArgsModel=TaskArguments,
        logger_name=logger.name,
    )
