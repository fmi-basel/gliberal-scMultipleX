# Copyright 2024 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Nicole Repina <nicole.repina@fmi.ch>
#

"""
Calculates spherical harmonics of input meshes.
"""
from typing import Any

import anndata as ad
import dask.array as da
import logging
import numpy as np
import os
import pandas as pd

import zarr
from vtkmodules.util import numpy_support
from fractal_tasks_core.tables import write_table
from fractal_tasks_core.tasks.io_models import InitArgsRegistrationConsensus
from pydantic.decorator import validate_arguments

from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.roi import (
    check_valid_ROI_indices,
    convert_indices_to_regions,
    convert_ROI_table_to_indices,
    load_region)

from scmultiplex.fractal.fractal_helper_functions import format_roi_table

from scmultiplex.meshing.MeshFunctions import export_stl_polydata, read_stl_polydata

from scmultiplex.aics_shparam.shparam import calculate_spherical_harmonics
from scmultiplex.aics_shparam import shtools

logger = logging.getLogger(__name__)


@validate_arguments
def spherical_harmonics_from_mesh(
        *,
        # Fractal arguments
        zarr_url: str,
        init_args: InitArgsRegistrationConsensus,
        # Task-specific arguments
        mesh_name: str = "org_linked_from_nuc",
        roi_table: str = "org_ROI_table_linked_3d",
        lmax: int = 2,
        translate_to_origin: bool = True,
        save_reconstructed_mesh: bool = True,
) -> dict[str, Any]:
    """
    Calculate spherical harmonics and reconstruction error of pre-computed meshes.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            Refers to the zarr_url of the reference acquisition.
            (standard argument for Fractal tasks, managed by Fractal server).
        init_args: Initialization arguments provided by
            `init_group_by_well_for_multiplexing`. It contains the
            zarr_url_list listing all the zarr_urls in the same well as the
            zarr_url of the reference acquisition that are being processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        mesh_name: Mesh folder name for which spherical harmonics are to be calculated
        roi_table: Name of the ROI table that corresponds to labels of meshed objects, e.g. org_ROI_table_3d
        lmax: Maximum degree of the spherical harmonics coefficients
        translate_to_origin: If true, translate centroid of mesh to origin prior to spherical harmonic decomposition.
            Recommended set to True
        save_reconstructed_mesh: If true, reconstruct mesh from spherical harmonics and save as stl in
            meshes zarr directory
    """
    logger.info(
        f"Running for {zarr_url=}. \n"
        f"Calculating spherical harmonics for meshes in {mesh_name=} with ROI table "
        f"{roi_table=}."
    )

    # Read ROIs of objects
    adata = ad.read_zarr(f"{zarr_url}/tables/{roi_table}")
    labels = adata.obs_vector('label')

    if len(labels) == 0:
        logger.warning("Well contains no objects")

    df_coeffs = []

    # for every object in ROI table...
    for row in adata.obs_names:
        row_int = int(row)
        org_label = labels[row_int]

        if not isinstance(org_label, str):
            raise TypeError('Label index must be string. Check ROI table obs naming.')

        mesh_fname = org_label + '.stl'
        mesh_path = f"{zarr_url}/meshes/{mesh_name}/{mesh_fname}"

        # Check that mesh for corresponding label id exists, if not continue to next id
        if os.path.isfile(mesh_path):
            mesh = read_stl_polydata(mesh_path)
        else:
            logger.warning(f"No mesh found for label {org_label}")
            continue

        ##############
        # Compute spherical harmonics coefficients  ###
        ##############

        coords = numpy_support.vtk_to_numpy(mesh.GetPoints().GetData())
        centroid = coords.mean(axis=0, keepdims=True)

        if translate_to_origin is True:
            # Translate to origin
            coords -= centroid
            mesh.GetPoints().SetData(numpy_support.numpy_to_vtk(coords))

            # # Equivalent way to transform mesh
            # x = coords[:, 0]
            # y = coords[:, 1]
            # z = coords[:, 2]
            # mesh = shtools.update_mesh_points(mesh, x, y, z)
            # # Calculate new centroid
            # coords = numpy_support.vtk_to_numpy(mesh.GetPoints().GetData())
            # centroid = coords.mean(axis=0, keepdims=True)

        # Calculate spherical harmonic decomposition
        # Note that meshes are not 2d aligned, orientation is as they are imaged in plate
        coeffs, grid_rec, grid_down = calculate_spherical_harmonics(mesh=mesh, lmax=lmax)

        # Calculate reconstruction error
        mse = shtools.get_reconstruction_error(grid_down, grid_rec)
        coeffs.update({'mse': mse, 'label': org_label})
        df_coeffs.append(coeffs)

        logger.info(f"Successfully calculated harmonics for mesh {mesh_fname}.")

        # Save mesh reconstructed from spherical harmonics
        if save_reconstructed_mesh:
            # Reconstruct mesh from grid
            mesh_rec = shtools.get_reconstruction_from_grid(grid_rec)
            save_transform_path = f"{zarr_url}/meshes/{mesh_name}_reconstructed"
            os.makedirs(save_transform_path, exist_ok=True)
            # save name is the organoid label id
            export_stl_polydata(os.path.join(save_transform_path, mesh_fname), mesh_rec)
            logger.info(f"Saved reconstructed mesh for mesh {mesh_fname}.")

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
        "fractal_table_version" : "1",
        "region": {"path": f"../meshes/{mesh_name}"},
        "instance_key": "label",
    }
    output_table_name = mesh_name + '_harmonics'
    write_table(
        image_group,
        output_table_name,
        measurement_table,
        overwrite=True,
        table_attrs=table_attrs,
    )
    logger.info(f"Successfully saved spherical harmonic coefficients as feature table {output_table_name}")
    logger.info(f"End spherical_harmonics_from_mesh task for {zarr_url=}")

    return {}


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=spherical_harmonics_from_mesh,
        logger_name=logger.name,
    )
