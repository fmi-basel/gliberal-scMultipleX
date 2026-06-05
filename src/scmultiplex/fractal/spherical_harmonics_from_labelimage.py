# Copyright 2024 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Nicole Repina <nicole.repina@fmi.ch>
#

"""
Calculate spherical harmonics of 3D input label image using aics_shparam.
"""

import logging
import os
from typing import Any

import anndata as ad
import dask.array as da
import numpy as np
import pandas as pd
import zarr
from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.roi import (
    check_valid_ROI_indices,
    convert_indices_to_regions,
    convert_ROI_table_to_indices,
    load_region,
)
from fractal_tasks_core.tables import write_table
from fractal_tasks_core.tasks.io_models import InitArgsRegistrationConsensus
from pydantic import validate_call

from scmultiplex.aics_shparam import shparam, shtools
from scmultiplex.fractal.fractal_helper_functions import format_roi_table, get_zattrs
from scmultiplex.meshing.MeshFunctions import export_stl_polydata

logger = logging.getLogger(__name__)


@validate_call
def spherical_harmonics_from_labelimage(
    *,
    # Fractal arguments
    zarr_url: str,
    init_args: InitArgsRegistrationConsensus,
    # Task-specific arguments
    label_name: str = "org_3d",
    roi_table: str = "org_ROI_table_3d",
    lmax: int = 2,
    save_mesh: bool = True,
    save_reconstructed_mesh: bool = True,
) -> dict[str, Any]:
    """
    Calculate spherical harmonics of 3D input label image using aics_shparam.

    This task consists of 6 parts:

    1. Load 3D label image based on provided label name and ROI table
    2. Compute 3D mesh with vtkContourFilter using aics_shparam functions
    3. Compute spherical harmonics of mesh using aics_shparam
    4. Compute reconstruction error (mse) of computed harmonics
    5. Optionally generate reconstructed mesh from the calculated harmonics
    6. Output: save the (1) spherical harmonic coefficients and mse error as feature table
        (2) reconstructed meshes (.stl) per object id in a new meshes folder within zarr structure

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            Refers to the zarr_url of the reference acquisition.
            (standard argument for Fractal tasks, managed by Fractal server).
        init_args: Intialization arguments provided by
            `init_group_by_well_for_multiplexing`. It contains the
            zarr_url_list listing all the zarr_urls in the same well as the
            zarr_url of the reference acquisition that are being processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        label_name: Label for which spherical harmonics are calculated
        roi_table: Name of the ROI table over which the task loops to
            calculate the registration. e.g. consensus object table 'org_ROI_table_consensus'
        lmax: Maximum degree of the spherical harmonics coefficients
        save_mesh: If True, saves the computed surface mesh (with vtkContourFilter in aics_shparam functions)
            on disk in subfolder 'meshes'. Filename corresponds to object label id
        save_reconstructed_mesh: If true, reconstruct mesh from spherical harmonics and save as stl in
            meshes zarr directory. Filename corresponds to object label id

    """
    logger.info(
        f"Running for {zarr_url=}. \n"
        f"Calculating spherical harmonics for objects in {label_name=} with ROI table "
        f"{roi_table=}."
    )

    # Lazily load zarr array for reference cycle
    # load well image as dask array e.g. for nuclear segmentation
    daska = da.from_zarr(f"{zarr_url}/labels/{label_name}/0")

    # Read ROIs of objects
    adata = ad.read_zarr(f"{zarr_url}/tables/{roi_table}")
    roi_attrs = get_zattrs(f"{zarr_url}/tables/{roi_table}")

    # Read Zarr metadata
    ngffmeta = load_NgffImageMeta(f"{zarr_url}/labels/{label_name}")
    xycoars = ngffmeta.coarsening_xy  # need to know when building new pyramids
    pixmeta = ngffmeta.get_pixel_sizes_zyx(level=0)

    # Create list of indices for 3D ROIs spanning the entire Z direction
    idlist = convert_ROI_table_to_indices(
        adata,
        level=0,
        coarsening_xy=xycoars,
        full_res_pxl_sizes_zyx=pixmeta,
    )

    check_valid_ROI_indices(idlist, roi_table)

    # Get labels to iterate over
    instance_key = roi_attrs["instance_key"]  # e.g. "label"

    # NGIO FIX, TEMP
    # Check that ROI_table.obs has the right column and extract label_value
    if instance_key not in adata.obs.columns:
        if adata.obs.index.name == instance_key:
            # Workaround for new ngio table
            adata.obs[instance_key] = adata.obs.index
        else:
            raise ValueError(
                f"In input ROI table, {instance_key=} "
                f" missing in {adata.obs.columns=}"
            )

    labels = adata.obs_vector(instance_key)

    # initialize variables
    compute = True  # convert to numpy array from dask

    if len(idlist) == 0:
        logger.warning("Well contains no objects")

    df_coeffs = []
    # for every object in ROI table...
    for i, obsname in enumerate(adata.obs_names):
        org_label = labels[i]
        region = convert_indices_to_regions(idlist[i])

        # load label image for object
        seg = load_region(
            data_zyx=daska,
            region=region,
            compute=compute,
        )

        # Mask to remove any neighboring organoids
        # TODO: Improve this during the NGIO refactor, very basic masking here to select the desired organoid label
        seg[seg != float(org_label)] = 0

        ##############
        # Compute mesh and spherical harmonics coefficients  ###
        ##############
        # Set spacing to ome-zarr pixel spacing metadata. Mesh will be in physical units (um)
        spacing = tuple(np.array(pixmeta))  # z,y,x e.g. (0.6, 0.216, 0.216)

        (
            (coeffs, grid_rec),
            (image_, mesh, grid_down, transform),
        ) = shparam.get_shcoeffs(image=seg, lmax=lmax, sigma=5, spacing=spacing)

        # Calculate reconstruction error
        mse = shtools.get_reconstruction_error(grid_down, grid_rec)
        coeffs.update({"mse": mse, "label": org_label})
        df_coeffs.append(coeffs)

        logger.info(f"Successfully calculated harmonics for label {org_label}.")

        ##############
        # Save mesh (optional) ###
        ##############

        # Save mesh calculated from label image as .stl
        if save_mesh:
            save_mesh_path = f"{zarr_url}/meshes/{roi_table}_shaics"
            os.makedirs(save_mesh_path, exist_ok=True)
            # save name is the organoid label id
            save_name = f"{int(org_label)}.stl"
            export_stl_polydata(os.path.join(save_mesh_path, save_name), mesh)
            logger.info(f"Saved surface mesh for object label {org_label}.")

        # Save mesh reconstructed from spherical harmonics as .stl
        if save_reconstructed_mesh:
            # Reconstruct mesh from grid
            mesh_rec = shtools.get_reconstruction_from_grid(grid_rec)
            save_transform_path = f"{zarr_url}/meshes/{roi_table}_shaics_reconstructed"
            os.makedirs(save_transform_path, exist_ok=True)
            # save name is the organoid label id
            save_name = f"{int(org_label)}.stl"
            export_stl_polydata(os.path.join(save_transform_path, save_name), mesh_rec)
            logger.info(f"Saved reconstructed mesh for object label {org_label}.")

    ##############
    # Save spherical harmonics as measurement table  ###
    ##############
    df_coeffs = pd.DataFrame(df_coeffs)

    # follows similar logic as feature extraction task
    if not df_coeffs.empty:
        measurement_table = format_roi_table([df_coeffs])
    else:
        # Create empty anndata table
        measurement_table = ad.AnnData()

    # Write to zarr group
    image_group = zarr.group(f"{zarr_url}")
    table_attrs = {
        "type": "feature_table",
        "fractal_table_version": "1",
        "region": {"path": f"../labels/{label_name}"},
        "instance_key": "label",
    }
    output_table_name = label_name + "_harmonics"
    write_table(
        image_group,
        output_table_name,
        measurement_table,
        overwrite=True,
        table_attrs=table_attrs,
    )

    logger.info(f"End spherical_harmonics_from_labelimage task for {zarr_url=}")

    return {}


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=spherical_harmonics_from_labelimage,
        logger_name=logger.name,
    )
