# Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Nicole Repina <nicole.repina@fmi.ch>
# Tommaso Comparin <tommaso.comparin@exact-lab.it>
# Joel Lüthi <joel.luethi@uzh.ch>
#

"""
Calculates translation for label-based registration
Note that this task is limited to well overviews that have same dimensions across multiplexing rounds
In case that dimensions differ, the padding performed in calculate_shift needs to be taken into account for shift values
Then apply_registration_to_ROI_tables task needs to be modified to accomodate cases of differing image dims
The minimum consensus region across rounds should be found, and this consensus can then be visualized in Napari
"""
import logging
from pathlib import Path
from typing import Any
from typing import Sequence

import anndata as ad
import dask.array as da
import numpy as np
import pandas as pd
import zarr
from anndata._io.specs import write_elem
from pydantic.decorator import validate_arguments
from skimage.registration import phase_cross_correlation

from fractal_tasks_core.lib_channels import get_channel_from_image_zarr
from fractal_tasks_core.lib_channels import OmeroChannel
from fractal_tasks_core.lib_ngff import load_NgffImageMeta
from fractal_tasks_core.lib_regions_of_interest import check_valid_ROI_indices
from fractal_tasks_core.lib_regions_of_interest import (
    convert_indices_to_regions,
)
from fractal_tasks_core.lib_regions_of_interest import (
    convert_ROI_table_to_indices,
)
from fractal_tasks_core.lib_regions_of_interest import load_region

from scmultiplex.linking.OrganoidLinkingFunctions import calculate_shift

logger = logging.getLogger(__name__)


@validate_arguments
def calculate_registration_label_based(
    *,
    # Fractal arguments
    input_paths: Sequence[str],
    output_path: str,
    component: str,
    metadata: dict[str, Any],
    # Task-specific arguments
    label_name: str,
    roi_table: str = "well_ROI_table",
    reference_cycle: int = 0,
    level: int = 2,
) -> dict[str, Any]:
    """
    Calculate registration based on images

    This task consists of 3 parts:

    1. Loading the images of a given ROI (=> loop over ROIs)
    2. Calculating the transformation for that ROI
    3. Storing the calculated transformation in the ROI table

    Parallelization level: image

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
        metadata: This parameter is not used by this task.
            (standard argument for Fractal tasks, managed by Fractal server).
        label_name: Label name that will be used for label-based
            registration; e.g. `org` from object segmentation.
        roi_table: Name of the ROI table over which the task loops to
            calculate the registration. Examples: `FOV_ROI_table` => loop over
            the field of views, `well_ROI_table` => process the whole well as
            one image.
        reference_cycle: Which cycle to register against. Defaults to 0,
            which is the first OME-Zarr image in the well (usually the first
            cycle that was provided).
        level: Pyramid level of the image to be segmented. Choose `0` to
            process at full resolution.

    """
    logger.info(
        f"Running for {input_paths=}, {component=}. \n"
        f"Calculating translation registration per {roi_table=} for "
        f"{label_name=}."
    )
    # Set OME-Zarr paths
    zarr_img_cycle_x = Path(input_paths[0]) / component

    # If the task is run for the reference cycle, exit
    # TODO: Improve the input for this: Can we filter components to not
    # run for itself?
    alignment_cycle = zarr_img_cycle_x.name
    if alignment_cycle == str(reference_cycle):
        logger.info(
            "Calculate registration image-based is running for "
            f"cycle {alignment_cycle}, which is the reference_cycle."
            "Thus, exiting the task."
        )
        return {}
    else:
        logger.info(
            "Calculate registration image-based is running for "
            f"cycle {alignment_cycle}"
        )

    # path to zarr file
    zarr_img_ref_cycle = zarr_img_cycle_x.parent / str(reference_cycle)

    # Read some parameters from Zarr metadata
    ngff_image_meta = load_NgffImageMeta(str(zarr_img_ref_cycle))
    coarsening_xy = ngff_image_meta.coarsening_xy

    # Lazily load zarr array
    # Reference (e.g. R0, fixed) vs. alignment (e.g. RX, moving)
    # load well image as dask array
    data_reference_zyx = da.from_zarr(f"{zarr_img_ref_cycle}/labels/{label_name}/{level}")
    data_alignment_zyx = da.from_zarr(f"{zarr_img_cycle_x}/labels/{label_name}/{level}")

    # Read ROIs
    ROI_table_ref = ad.read_zarr(f"{zarr_img_ref_cycle}/tables/{roi_table}")
    # TODO: Nicole thinks there is an error in fractal_core function here; load zarr table of x!
    # TODO: Joel add to your calculate_registration_image_based code!
    ROI_table_x = ad.read_zarr(f"{zarr_img_cycle_x}/tables/{roi_table}")
    logger.info(
        f"Found {len(ROI_table_x)} ROIs in {roi_table=} to be processed."
    )

    # Read pixel sizes from zarr attributes
    ngff_image_meta_cycle_x = load_NgffImageMeta(str(zarr_img_cycle_x))
    pxl_sizes_zyx = ngff_image_meta.get_pixel_sizes_zyx(level=0)
    pxl_sizes_zyx_cycle_x = ngff_image_meta_cycle_x.get_pixel_sizes_zyx(
        level=0
    )

    if pxl_sizes_zyx != pxl_sizes_zyx_cycle_x:
        raise ValueError(
            "Pixel sizes need to be equal between cycles for registration"
        )

    # Create list of indices for 3D ROIs spanning the entire Z direction
    list_indices_ref = convert_ROI_table_to_indices(
        ROI_table_ref,
        level=level,
        coarsening_xy=coarsening_xy,
        full_res_pxl_sizes_zyx=pxl_sizes_zyx,
    )
    check_valid_ROI_indices(list_indices_ref, roi_table)

    list_indices_cycle_x = convert_ROI_table_to_indices(
        ROI_table_x,
        level=level,
        coarsening_xy=coarsening_xy,
        full_res_pxl_sizes_zyx=pxl_sizes_zyx,
    )
    check_valid_ROI_indices(list_indices_cycle_x, roi_table)

    num_ROIs = len(list_indices_ref)
    compute = True
    new_shifts = {}
    for i_ROI in range(num_ROIs):
        logger.info(
            f"Now processing ROI {i_ROI+1}/{num_ROIs} "
            f"for label {label_name}."
        )
        img_ref = load_region(
            data_zyx=data_reference_zyx,
            region=convert_indices_to_regions(list_indices_ref[i_ROI]),
            compute=compute,
        )
        img_cycle_x = load_region(
            data_zyx=data_alignment_zyx,
            region=convert_indices_to_regions(list_indices_cycle_x[i_ROI]),
            compute=compute,
        )

        ##############
        #  Calculate the transformation
        ##############

        # calculate shifts on padded images that have the same shape (pad performed within calculate_shift function)
        shifts, _, _ = calculate_shift(np.squeeze(img_ref), np.squeeze(img_cycle_x), bin=1)
        
        logger.info(shifts)

        ##############
        # Storing the calculated transformation ###
        ##############
        # Store the shift in ROI table
        # TODO: Store in OME-NGFF transformations: Check SpatialData approach,
        # per ROI storage?

        # Adapt ROIs for the given ROI table:
        ROI_name = ROI_table_ref.obs.index[i_ROI]
        new_shifts[ROI_name] = calculate_physical_shifts(
            shifts,
            level=level,
            coarsening_xy=coarsening_xy,
            full_res_pxl_sizes_zyx=pxl_sizes_zyx,
        )

    # Write physical shifts to disk (as part of the ROI table, add additional translation xyz columns)
    # Update the original well_ROI_table with shifts
    logger.info(f"Updating the {roi_table=} with translation columns")
    new_ROI_table = get_ROI_table_with_translation(ROI_table_x, new_shifts)
    group_tables = zarr.group(f"{zarr_img_cycle_x}/tables/")
    write_elem(group_tables, roi_table, new_ROI_table)
    group_tables[roi_table].attrs["type"] = "ngff:region_table"

    return {}


def calculate_physical_shifts(
    shifts: np.array,
    level: int,
    coarsening_xy: int,
    full_res_pxl_sizes_zyx: list[float],
) -> list[float]:
    """
    Calculates shifts in physical units based on pixel shifts

    Args:
        shifts: array of shifts, zyx or yx
        level: resolution level
        coarsening_xy: coarsening factor between levels
        full_res_pxl_sizes_zyx: pixel sizes in physical units as zyx

    Returns:
        shifts_physical: shifts in physical units as zyx
    """
    curr_pixel_size = np.array(full_res_pxl_sizes_zyx) * coarsening_xy**level
    if len(shifts) == 3:
        shifts_physical = shifts * curr_pixel_size
    elif len(shifts) == 2:
        shifts_physical = [
            0,
            shifts[0] * curr_pixel_size[1],
            shifts[1] * curr_pixel_size[2],
        ]
    else:
        raise ValueError(
            f"Wrong input for calculate_physical_shifts ({shifts=})"
        )
    return shifts_physical


def get_ROI_table_with_translation(
    ROI_table: ad.AnnData,
    new_shifts: dict[str, list[float]],
) -> ad.AnnData:
    """
    Adds translation columns to a ROI table

    Args:
        ROI_table: Fractal ROI table
        new_shifts: zyx list of shifts

    Returns:
        Fractal ROI table with 3 additional columns for calculated translations
    """

    shift_table = pd.DataFrame(new_shifts).T
    translation_columns = ["translation_z", "translation_y", "translation_x"]
    shift_table.columns = translation_columns
    shift_table = shift_table.rename_axis("FieldIndex")

    # check if translation columns already exist; if exists, rewrite instead of doing a merge
    new_roi_table = ROI_table.to_df()
    # TODO: Joel add to your calculate_registration_image_based code!
    if set(translation_columns).issubset(new_roi_table.columns):
        logger.info(
            "Columns are present : Yes "
            "Overwriting translation columns"
        )
        new_roi_table[translation_columns] = shift_table[translation_columns]

    else:
        logger.info(
            "Columns are present : No "
            "Adding new translation columns"
        )
        new_roi_table = new_roi_table.merge(
            shift_table, left_index=True, right_index=True
        )

    if len(new_roi_table) != len(ROI_table):
        raise ValueError(
            "New ROI table with registration info has a "
            f"different length ({len(new_roi_table)=}) "
            f"from the original ROI table ({len(ROI_table)=})"
        )

    positional_columns = [
        "x_micrometer",
        "y_micrometer",
        "z_micrometer",
        "len_x_micrometer",
        "len_y_micrometer",
        "len_z_micrometer",
        # "x_micrometer_original",
        # "y_micrometer_original",
        "translation_z",
        "translation_y",
        "translation_x",
    ]

    adata = ad.AnnData(
        X=new_roi_table.loc[:, positional_columns].astype(np.float32)
    )
    adata.obs_names = new_roi_table.index
    adata.var_names = list(map(str, positional_columns))
    return adata


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=calculate_registration_image_based,
        logger_name=logger.name,
    )

    
    


