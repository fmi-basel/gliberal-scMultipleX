# Copyright 2024 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Nicole Repina <nicole.repina@fmi.ch>
#

"""
Extract shape features of 3D input meshes.
"""

import logging
import os
from typing import Any

import anndata as ad
import pandas as pd
import zarr
from fractal_tasks_core.tables import write_table
from fractal_tasks_core.tasks.io_models import InitArgsRegistrationConsensus
from pydantic import validate_call
from vtkmodules.util import numpy_support

from scmultiplex.aics_shparam import shtools
from scmultiplex.aics_shparam.shparam import calculate_spherical_harmonics
from scmultiplex.features.MeshExtraction import get_mesh_measurements
from scmultiplex.fractal.fractal_helper_functions import format_roi_table, get_zattrs
from scmultiplex.meshing.MeshFunctions import (
    export_stl_polydata,
    export_vtk_polydata,
    get_gaussian_curvatures,
    read_stl_polydata,
)

logger = logging.getLogger(__name__)


@validate_call
def scmultiplex_mesh_measurements(
    *,
    # Fractal arguments
    zarr_url: str,
    init_args: InitArgsRegistrationConsensus,
    # Task-specific arguments
    mesh_name: str,
    roi_table: str,
    output_table_name: str,
    save_hulls: bool = True,
    calculate_curvature: bool = True,
    calculate_harmonics: bool = True,
    lmax: int = 2,
    translate_to_origin: bool = True,
    save_reconstructed_mesh: bool = True,
) -> dict[str, Any]:
    """
    Extract shape features of 3D input meshes.

    This task consists of 5 parts:

    1. Load meshes (.stl) from zarr structure that have been previously generated
        (e.g. using Surface Mesh Multiscale task)
    2. Extract mesh morphology measurements: volume, surface area, extent, solidity, concavity, asymmetry,
        aspect ratio, and normalized surface area to volume ratio. Units correspond to units of mesh. If mesh was
        generated with surface_mesh_multiscale task, units are physical units (um). For further details of features see
        scMultiplex FeatureFunctions.py
    3. Optionally calculate Gaussian curvature at each mesh point (set calculate_curvature = True)
    4. Optionally compute spherical harmonics of mesh using aics_shparam (set calculate_harmonics = True).
        Compute reconstruction error (mse) of computed harmonics and optionally generate reconstructed mesh from the
        calculated harmonics
    5. Output: save (1) extracted measurements as a new table (output_table_name), (2) optional spherical harmonic
        coefficients and mse error as a separate feature table ('output_table_name'_harmonics) (3) optional
        reconstructed meshes from spherical harmonics as .stl within zarr structure (3) optional convex hull
        and bounding box on disk as .vtp within zarr structure (4) optional curvature meshes on disk as .vtp within
        zarr structure.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            Refers to the zarr_url of the reference acquisition.
            (standard argument for Fractal tasks, managed by Fractal server).
        init_args: Initialization arguments provided by
            `init_group_by_well_for_multiplexing`. It contains the
            zarr_url_list listing all the zarr_urls in the same well as the
            zarr_url of the reference acquisition that are being processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        mesh_name: Mesh folder name for which features are to be extracted. Must contain .stl format meshes whose
            filename corresponds to label name in ROI table.
        roi_table: Name of the ROI table that corresponds to labels of meshed objects, only used for indexing
            objects
        output_table_name: Name of the output AnnData table to save the
            measurements in. A table of this name can't exist yet in the
            OME-Zarr file
        save_hulls: if True, save the calculated convex hull and bounding box as .vtp meshes within
            meshes/[mesh_name]_convex_hull and meshes/[mesh_name]_bounding_box directories
        calculate_curvature: if True, calculate Gaussian curvature at each mesh point and save as .vtp mesh
            on disk within meshes/[mesh_name]_curvature folder in zarr structure. Filename
            corresponds to object label id.
        calculate_harmonics: if True, calculate spherical harmonics of mesh using aics_shparam
        lmax: Maximum degree of the spherical harmonics coefficients
        translate_to_origin: If true, translate centroid of mesh to origin prior to spherical harmonic decomposition.
            Recommended set to True
        save_reconstructed_mesh: If true, reconstruct mesh from spherical harmonics and save as stl in
            meshes zarr directory

    """
    logger.info(
        f"Running for {zarr_url=}. \n"
        f"Extracting features for meshes in {mesh_name=} with ROI table "
        f"{roi_table=}."
    )

    # Read ROIs of objects
    adata = ad.read_zarr(f"{zarr_url}/tables/{roi_table}")
    roi_attrs = get_zattrs(f"{zarr_url}/tables/{roi_table}")
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

    if len(labels) == 0:
        logger.warning("Well contains no objects")

    df_feats = []
    df_coeffs = []

    # TODO with NGIO refactor, consider not looping over ROI table but directly over mesh names. This will
    #  generalize task to not require a corresponding ROI table in case meshes were generated outside of Fractal
    # for every object in ROI table...
    for i, obsname in enumerate(adata.obs_names):
        org_label = labels[i]

        if not isinstance(org_label, str):
            raise TypeError("Label index must be string. Check ROI table obs naming.")

        mesh_fname = org_label + ".stl"
        mesh_path = f"{zarr_url}/meshes/{mesh_name}/{mesh_fname}"

        # Check that mesh for corresponding label id exists, if not continue to next id
        if os.path.isfile(mesh_path):
            polydata = read_stl_polydata(mesh_path)
        else:
            logger.warning(f"No mesh found for label {org_label}")
            continue

        ##############
        # Extract features  ###
        ##############
        measurements = {"label": org_label}
        (
            vtk_measurements,
            convex_hull_polydata,
            bounding_box_polydata,
        ) = get_mesh_measurements(polydata)

        logger.info(f"Successfully extracted features for mesh {mesh_fname}.")

        ##############
        # Combine measurements and save polydata ###
        ##############
        measurements.update(vtk_measurements)
        df_feats.append(measurements)

        if save_hulls:
            # save convex hull
            save_transform_path = f"{zarr_url}/meshes/{mesh_name}_convex_hull"
            os.makedirs(save_transform_path, exist_ok=True)
            # Save name is the organoid label id
            save_name = f"{int(org_label)}.vtp"
            export_vtk_polydata(
                os.path.join(save_transform_path, save_name), convex_hull_polydata
            )

            # save bounding box
            save_transform_path = f"{zarr_url}/meshes/{mesh_name}_bounding_box"
            os.makedirs(save_transform_path, exist_ok=True)
            export_vtk_polydata(
                os.path.join(save_transform_path, save_name), bounding_box_polydata
            )

        if calculate_curvature:
            # Calculate curvature
            polydata_curv, scalar_range, curvatures_numpy = get_gaussian_curvatures(
                polydata
            )
            # Save mesh
            save_transform_path = f"{zarr_url}/meshes/{mesh_name}_curvature"
            os.makedirs(save_transform_path, exist_ok=True)
            # Save name is the organoid label id
            save_name_curv = f"{int(org_label)}.vtp"
            export_vtk_polydata(
                os.path.join(save_transform_path, save_name_curv), polydata_curv
            )

        ##############
        # Compute spherical harmonics coefficients  ###
        ##############

        if calculate_harmonics:
            # Compute spherical harmonics coefficients
            coords = numpy_support.vtk_to_numpy(polydata.GetPoints().GetData())
            centroid = coords.mean(axis=0, keepdims=True)

            if translate_to_origin is True:
                # Translate to origin
                coords -= centroid
                polydata.GetPoints().SetData(numpy_support.numpy_to_vtk(coords))

            # Calculate spherical harmonic decomposition
            # Note that meshes are not 2d aligned, orientation is as they are imaged in plate
            coeffs, grid_rec, grid_down = calculate_spherical_harmonics(
                mesh=polydata, lmax=lmax
            )

            # Calculate reconstruction error
            mse = shtools.get_reconstruction_error(grid_down, grid_rec)
            coeffs.update({"mse": mse, "label": org_label})
            df_coeffs.append(coeffs)

            logger.info(
                f"Successfully calculated spherical harmonics for mesh {mesh_fname}."
            )

            if save_reconstructed_mesh:
                # Reconstruct mesh from grid
                mesh_rec = shtools.get_reconstruction_from_grid(grid_rec)
                save_transform_path = f"{zarr_url}/meshes/{mesh_name}_reconstructed"
                os.makedirs(save_transform_path, exist_ok=True)
                # save name is the organoid label id
                export_stl_polydata(
                    os.path.join(save_transform_path, mesh_fname), mesh_rec
                )
                logger.info(
                    f"Saved reconstructed mesh from harmonics for mesh {mesh_fname}."
                )

    ##############
    # Save extracted features as measurement table  ###
    ##############
    df_feats = pd.DataFrame(df_feats)

    # follows similar logic as feature extraction task
    if not df_feats.empty:
        measurement_table = format_roi_table([df_feats])
    else:
        # Create empty anndata table
        measurement_table = ad.AnnData()

    # Write to zarr group
    image_group = zarr.group(f"{zarr_url}")
    table_attrs = {
        "type": "feature_table",
        "fractal_table_version": "1",
        "region": {"path": f"../meshes/{mesh_name}"},
        "instance_key": "label",
    }

    write_table(
        image_group,
        output_table_name,
        measurement_table,
        overwrite=True,
        table_attrs=table_attrs,
    )
    logger.info(
        f"Successfully saved extracted features as feature table {output_table_name}"
    )

    ##############
    # Save spherical harmonics as measurement table  ###
    ##############
    if calculate_harmonics:
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
            "region": {"path": f"../meshes/{mesh_name}"},
            "instance_key": "label",
        }

        output_table_harmonics_name = output_table_name + "_harmonics"
        write_table(
            image_group,
            output_table_harmonics_name,
            measurement_table,
            overwrite=True,
            table_attrs=table_attrs,
        )
        logger.info(
            f"Successfully saved spherical harmonic coefficients as feature table "
            f"{output_table_harmonics_name}"
        )

    logger.info(f"End scmultiplex_mesh_measurements task for {zarr_url=}")

    return {}


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=scmultiplex_mesh_measurements,
        logger_name=logger.name,
    )
