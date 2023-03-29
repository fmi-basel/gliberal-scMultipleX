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
from skimage.measure import regionprops
from scmultiplex.features.FeatureFunctions import (
    fixed_percentiles,
    kurtos,
    skewness,
    stdv,
    disconnected_component,
    surface_area_marchingcube,
)

from scmultiplex.features.FeatureFunctions import (
    minor_major_axis_ratio,
    convex_hull_area_resid,
    convex_hull_centroid_dif,
    circularity,
    aspect_ratio,
    concavity_count,
)

import fractal_tasks_core
from fractal_tasks_core.lib_channels import get_channel_from_image_zarr
from fractal_tasks_core.lib_regions_of_interest import (
    convert_ROI_table_to_indices,
)
from fractal_tasks_core.lib_upscale_array import upscale_array
from fractal_tasks_core.lib_zattrs_utils import extract_zyx_pixel_sizes


__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__


logger = logging.getLogger(__name__)


def get_regionprops_measurements(regionproperties, channel_prefix: str = ""):
    row_list = []
    for labeled_obj in regionproperties:
        common_row = {
            "label": int(labeled_obj["label"]),
            "mean_intensity": labeled_obj["mean_intensity"],
            "max_intensity": labeled_obj["max_intensity"],
            "min_intensity": labeled_obj["min_intensity"],
            "percentile25": labeled_obj["fixed_percentiles"][0],
            "percentile50": labeled_obj["fixed_percentiles"][1],
            "percentile75": labeled_obj["fixed_percentiles"][2],
            "percentile90": labeled_obj["fixed_percentiles"][3],
            "percentile95": labeled_obj["fixed_percentiles"][4],
            "percentile99": labeled_obj["fixed_percentiles"][5],
            "stdev": labeled_obj["stdv"],
            "skew": labeled_obj["skewness"],
            "kurtosis": labeled_obj["kurtos"],
        }

        # Additional features
        # 'imgdim_x': img_dim[1],
        # 'imgdim_y': img_dim[0],
        # 'x_pos_pix': labeled_obj['centroid'][1],
        # 'y_pos_pix': labeled_obj['centroid'][0],
        # 'x_pos_weighted_pix': labeled_obj['weighted_centroid'][1],
        # 'y_pos_weighted_pix': labeled_obj['weighted_centroid'][0],
        # 'x_massDisp_pix': labeled_obj['weighted_centroid'][1] - labeled_obj['centroid'][1],
        # 'y_massDisp_pix': labeled_obj['weighted_centroid'][0] - labeled_obj['centroid'][0],
        # 'abs_min': abs_min_intensity,
        # 'area_pix': labeled_obj['area'],
        # 'area_convhull': labeled_obj['area_convex'],
        # 'area_bbox': labeled_obj['area_bbox'],
        # 'perimeter': labeled_obj['perimeter'],
        # 'equivDiam': labeled_obj['equivalent_diameter_area'],
        # 'eccentricity': labeled_obj['eccentricity'],
        # 'circularity': circularity(labeled_obj),
        # 'solidity': labeled_obj['solidity'],
        # 'extent': labeled_obj['extent'],
        # 'majorAxisLength': labeled_obj['major_axis_length'],
        # 'minorAxisLength': labeled_obj['minor_axis_length'],
        # 'minmajAxisRatio': minor_major_axis_ratio(labeled_obj),
        # 'aspectRatio': aspect_ratio(labeled_obj),
        # 'concavity': convex_hull_area_resid(labeled_obj),
        # 'concavity_count': concavity_count(labeled_obj, min_area_fraction=0.005),
        # 'asymmetry': convex_hull_centroid_dif(labeled_obj),

        row_list.append(common_row)
    df_well = pd.DataFrame(row_list)

    # FIXME: Add the prefix only to relevant columns (i.e. not label, not shapes etc.)
    df_well = df_well.add_prefix(channel_prefix + ".")
    df_well["label"] = df_well[channel_prefix + ".label"]
    return df_well


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
    level = 0
):
    """
    Wrapper task for scmultiplex measurements for Fractal

    :param input_paths: TBD (default arg for Fractal tasks)
    :param output_path: TBD (default arg for Fractal tasks)
    :param metadata: TBD (default arg for Fractal tasks)
    :param component: TBD (default arg for Fractal tasks)
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
    num_levels = metadata["num_levels"]
    coarsening_xy = metadata["coarsening_xy"]
    label_dtype = np.uint32

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
    print(spacing)

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
        print(name, params)
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
        # FIXME: Stop input_channels is empty? Or just make shape measurements?
        logger.warning(f"No images loaded for {input_channels=}")

        # upscale_labels = True
    # Loop over label inputs and load corresponding (upscaled) image

    # FIXME: Add check whether label exists?
    input_label_image = da.from_zarr(
        f"{in_path}/{component}/labels/{label_image}/{level}"
    )

    # FIXME: Allow shape measurements without intensity image?
    # Then need to upscale differently
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
    for i_ROI, indices in enumerate(list_indices):
        s_z, e_z, s_y, e_y, s_x, e_x = indices[:]
        region = (slice(s_z, e_z), slice(s_y, e_y), slice(s_x, e_x))

        logger.info(f"ROI {i_ROI+1}/{num_ROIs}: {region=}")

        # Load the label image
        label_img = input_label_image[region].compute()

        # FIXME: Get actual spacing
        spacing = (1, 1, 1)

        # Set inputs
        df_roi = pd.DataFrame()
        for input_name in input_channels.keys():
            img = input_image_arrays[input_name][region].compute()

            # TODO: Use different measurements for 2D vs. 3D
            # shape = input_label_arrays[name].shape
            # if shape[0] == 1:

            # TODO: If it's 2D, does it make a difference whether the image is (1, 2160, 2560) vs. (2160, 2560)?
            # Squash the first axis if only 1 there? And then also skip the first spacing axis if they are 3?

            regionproperties = regionprops(
                label_img,
                img,
                extra_properties=(
                    fixed_percentiles,
                    skewness,
                    kurtos,
                    stdv,
                    surface_area_marchingcube,
                ),
                spacing=spacing,
            )
            new_df = get_regionprops_measurements(
                regionproperties, channel_prefix=input_name
            )

            if "label" in df_roi.columns:
                df_roi = df_roi.merge(right=new_df, how="outer", on="label")
            else:
                df_roi = pd.concat([df_roi, new_df], axis=1)

        df_well = pd.concat([df_well, df_roi], axis=0, ignore_index=True)

    print(df_well)
    print(list(df_well.columns))



    # Output handling: "dataframe" type (for each output, concatenate ROI
    # dataframes, clean up, and store in a AnnData table on-disk)

    # Concatenate all FOV dataframes
    # list_dfs = output_dataframe_lists[name]
    # df_well = pd.concat(list_dfs, axis=0, ignore_index=True)
    # Extract labels and drop them from df_well

    labels = pd.DataFrame(df_well["label"].astype(str))
    df_well.drop(labels=["label"], axis=1, inplace=True)
    # Convert all to float (warning: some would be int, in principle)
    measurement_dtype = np.float32
    df_well = df_well.astype(measurement_dtype)
    # Convert to anndata
    measurement_table = ad.AnnData(df_well, dtype=measurement_dtype)
    measurement_table.obs = labels
    # Write to zarr group
    group_tables = zarr.group(f"{in_path}/{component}/tables/")
    write_elem(group_tables, output_table_name, measurement_table)
    # Update OME-NGFF metadata
    current_tables = group_tables.attrs.asdict().get("tables") or []
    if output_table_name in current_tables:
        # FIXME: move this check to an earlier stage of the task
        raise ValueError(
            f"{in_path}/{component}/tables/ already includes "
            f"{output_table_name=} in {current_tables=}"
        )
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

    run_fractal_task(
        task_function=scmultiplex_measurements,
        TaskArgsModel=TaskArguments,
        logger_name=logger.name,
    )
