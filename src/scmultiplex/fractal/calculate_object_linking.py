# Copyright 2023 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Nicole Repina <nicole.repina@fmi.ch>
# Tommaso Comparin <tommaso.comparin@exact-lab.it>
# Joel Lüthi <joel.luethi@uzh.ch>
#

"""
Calculates linking tables for segmented objects in a well
"""
import logging
from typing import Any, Optional

import anndata as ad
import dask.array as da
import numpy as np
import zarr
from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.roi import (
    check_valid_ROI_indices,
    convert_indices_to_regions,
    convert_ROI_table_to_indices,
    load_region,
)
from fractal_tasks_core.tables import write_table
from fractal_tasks_core.tasks.io_models import InitArgsRegistration
from pydantic import validate_call

from scmultiplex.fractal.fractal_helper_functions import (
    calculate_physical_shifts,
    extract_acq_info,
    get_ROI_table_with_translation,
    get_zattrs,
)
from scmultiplex.linking.OrganoidLinkingFunctions import (
    apply_shift,
    calculate_matching,
    calculate_shift,
)

logger = logging.getLogger(__name__)


@validate_call
def calculate_object_linking(
    *,
    # Fractal arguments
    zarr_url: str,
    init_args: InitArgsRegistration,
    # Task-specific arguments
    label_name: str,
    roi_table: str = "well_ROI_table",
    new_linking_table_name: Optional[str] = None,
    level: int = 0,
    iou_cutoff: float = 0.2,
) -> dict[str, Any]:
    """
    Calculate object linking based on segmentation label map images

    This task consists of 4 parts:

    1. Load the object segmentation images for each well (paired reference and alignment round)
    2. Calculate the shift transformation for the image pair
    3. Apply shifts to image pair and identify matching object labels given an iou cutoff threshold
    4. Store the identified matches as a linking table in alignment round directory
    5. Update input ROI table to store shifts for use in subsequent tasks. Shifts stored in physical
        coordinates in column names translation_x, _y, _z in ROI table.

    Parallelization level: image

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        init_args: Intialization arguments provided by
            `_image_based_registration_hcs_init`. They contain the
            reference_zarr_url that is used for registration.
            (standard argument for Fractal tasks, managed by Fractal server).
        label_name: Label name that will be used for label-based
            registration; e.g. `org` from object segmentation.
        roi_table: Name of the well ROI table. Input ROI table must have single ROI entry;
            e.g. `well_ROI_table`
        new_linking_table_name: Optional new table name for linked matches, saved in RX
            alignment round directory.
            If left None, default is {label_name}_match_table
        level: Pyramid level of the image to be processed. Choose `0` to
            process at full resolution.
        iou_cutoff: Float in range 0 to 1 to specify intersection over union cutoff.
            Object pairs that have an iou below this value are filtered out and not stored in linking table.
    """

    logger.info(
        f"Running for {zarr_url=}\n"
        f"Calculating object linking with {roi_table=} for "
        f"{label_name=}."
    )
    # Set OME-Zarr paths
    r0_zarr_path = init_args.reference_zarr_url

    zarr_acquisition = extract_acq_info(zarr_url)
    ref_acquisition = extract_acq_info(r0_zarr_path)

    # Read Zarr metadata
    r0_ngffmeta = load_NgffImageMeta(r0_zarr_path)
    rx_ngffmeta = load_NgffImageMeta(zarr_url)
    r0_xycoars = r0_ngffmeta.coarsening_xy
    rx_xycoars = rx_ngffmeta.coarsening_xy
    r0_pixmeta = r0_ngffmeta.get_pixel_sizes_zyx(level=0)
    rx_pixmeta = rx_ngffmeta.get_pixel_sizes_zyx(level=0)

    if r0_pixmeta != rx_pixmeta:
        raise ValueError("Pixel sizes need to be equal between cycles for registration")

    # Lazily load zarr array
    # Reference (e.g. R0, fixed) vs. alignment (e.g. RX, moving)
    # load well image as dask array
    r0_dask = da.from_zarr(f"{r0_zarr_path}/labels/{label_name}/{level}")
    rx_dask = da.from_zarr(f"{zarr_url}/labels/{label_name}/{level}")

    # Read ROIs
    r0_adata = ad.read_zarr(f"{r0_zarr_path}/tables/{roi_table}")
    rx_adata = ad.read_zarr(f"{zarr_url}/tables/{roi_table}")
    rx_adata_attrs = get_zattrs(f"{zarr_url}/tables/{roi_table}")
    logger.info(f"Found {len(rx_adata)} ROIs in {roi_table=} to be processed.")

    # Create list of indices for 3D ROIs spanning the entire Z direction
    r0_idlist = convert_ROI_table_to_indices(
        r0_adata,
        level=level,
        coarsening_xy=r0_xycoars,
        full_res_pxl_sizes_zyx=r0_pixmeta,
    )
    check_valid_ROI_indices(r0_idlist, roi_table)

    rx_idlist = convert_ROI_table_to_indices(
        rx_adata,
        level=level,
        coarsening_xy=rx_xycoars,
        full_res_pxl_sizes_zyx=rx_pixmeta,
    )
    check_valid_ROI_indices(rx_idlist, roi_table)

    if len(r0_idlist) > 1 or len(rx_idlist) > 1:
        raise ValueError("Well overview must contain single region of interest")

    if len(r0_idlist) == 0 or len(rx_idlist) == 0:
        raise ValueError("Well overview ROI is empty")

    ##############
    #  Calculate the transformation
    ##############

    # initialize variables
    compute = True

    # TODO note that with [0] assume that ROI table has single ROI, i.e. it is a well overview.
    #  Consider generalizing to multi-ROI processing, e.g. for FOV ROI tables
    r0 = load_region(
        data_zyx=r0_dask,
        region=convert_indices_to_regions(r0_idlist[0]),
        compute=compute,
    )
    rx = load_region(
        data_zyx=rx_dask,
        region=convert_indices_to_regions(rx_idlist[0]),
        compute=compute,
    )

    r0 = np.squeeze(r0)
    rx = np.squeeze(rx)
    # calculate shifts on padded images that have the same shape (pad performed within calculate_shift function)
    shifts, r0_pad, rx_pad = calculate_shift(r0, rx, bin=1)

    logger.info(shifts)

    # shift the rx image to match r0 image
    rx_pad_shifted = apply_shift(rx_pad, shifts)

    # run matching
    # column names of link_df are ["R0_label", "RX_label", "iou"],
    stat, link_df_unfiltered, link_df = calculate_matching(
        r0_pad, rx_pad_shifted, iou_cutoff
    )

    # log matching output
    logger.info(
        f"{stat[2]} out of {stat[10]} alignment objects are NOT matched to a reference object."
    )
    logger.info(
        f"{stat[4]} out of {stat[9]} reference objects are NOT matched to an alignment object."
    )
    logger.info(
        f"removed {len(link_df_unfiltered) - len(link_df)} out of {len(link_df_unfiltered)} alignment "
        f"objects that are not matched to reference."
    )

    # format output df and convert to anndata
    link_df = link_df.sort_values(by=["R0_label"])
    link_df = link_df.rename(
        columns={
            "R0_label": "R" + str(ref_acquisition) + "_label",
            "RX_label": "R" + str(zarr_acquisition) + "_label",
        }
    )
    logger.info(link_df)

    # TODO refactor into helper function
    link_df_adata = ad.AnnData(X=np.array(link_df, dtype=np.float32))
    obsnames = list(map(str, link_df.index))
    varnames = list(link_df.columns.values)
    link_df_adata.obs_names = obsnames
    link_df_adata.var_names = varnames

    ##############
    # Storing the calculated transformation in rx directory ###
    ##############

    # Generate linking table: 3 columns ["R0_label", "RX_label", "iou"], where are RX is R1,R2, etc
    if new_linking_table_name is None:
        new_link_table = label_name + "_match_table"
    else:
        new_link_table = new_linking_table_name

    # Save the shifted linking table as a new table
    logger.info(
        f"Write the registered ROI table {new_link_table} for round {zarr_acquisition}"
    )

    image_group = zarr.group(f"{zarr_url}")

    write_table(
        image_group,
        new_link_table,
        link_df_adata,
        overwrite=True,
        table_attrs=dict(type="linking_table", fractal_table_version="1"),
    )

    # write new well_ROI_table; remains unmodified except add
    # shifts to roi columns as translation_x, translation_y and translation_z
    roi_name = rx_adata.obs.index[0]
    shifts = np.insert(shifts, 0, 0)  # insert 0 at index 0; convert to zyx from yx
    # convert shifts from pixels to physical coordinates
    new_shifts = {
        roi_name: calculate_physical_shifts(
            shifts,
            level=level,
            coarsening_xy=rx_xycoars,
            full_res_pxl_sizes_zyx=rx_pixmeta,
        )
    }

    # Write physical shifts to disk (as part of the ROI table)
    logger.info(f"Updating the {roi_table=} with translation columns")
    image_group = zarr.group(zarr_url)
    new_ROI_table = get_ROI_table_with_translation(rx_adata, new_shifts)
    write_table(
        image_group,
        roi_table,
        new_ROI_table,
        overwrite=True,
        table_attrs=rx_adata_attrs,
    )


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=calculate_object_linking,
        logger_name=logger.name,
    )
