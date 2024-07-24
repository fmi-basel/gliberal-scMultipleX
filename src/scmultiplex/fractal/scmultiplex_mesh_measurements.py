# Copyright 2024 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Nicole Repina <nicole.repina@fmi.ch>
#

"""
Extract shape features of 3D input meshes.
"""

import anndata as ad
import logging
import os
import pandas as pd
import zarr

from fractal_tasks_core.tables import write_table
from fractal_tasks_core.tasks.io_models import InitArgsRegistrationConsensus
from pydantic.decorator import validate_arguments
from typing import Any

from scmultiplex.features.MeshExtraction import get_mesh_measurements
from scmultiplex.fractal.fractal_helper_functions import format_roi_table
from scmultiplex.meshing.MeshFunctions import read_stl_polydata, get_gaussian_curvatures, export_vtk_polydata

logger = logging.getLogger(__name__)


@validate_arguments
def scmultiplex_mesh_measurements(
        *,
        # Fractal arguments
        zarr_url: str,
        init_args: InitArgsRegistrationConsensus,
        # Task-specific arguments
        mesh_name: str,
        roi_table: str,
        output_table_name: str,
        calculate_curvature: bool = True,
        save_hulls: bool = True,
) -> dict[str, Any]:
    """
    Extract shape features of 3D input meshes.

    This task consists of 3 parts:

    1. Load meshes from specified mesh folder name within zarr structure
    2. Extract mesh morphology measurements: volume, surface area, extent, solidity, concavity, asymmetry,
        aspect ratio, and normalized surface area to volume ratio. Units correspond to units of mesh. If mesh was
        generated with surface_mesh_multiscale task, units are physical units (um). For further details of features see
        scMultiplex FeatureFunctions.py
    3. Save extracted measurements as a new table (output_table_name), and optionally save
        the calculated convex hull, bounding box, and curvature meshes on disk.

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
        calculate_curvature: if True, calculate Gaussian curvature at each mesh point and save as .vtp mesh
            on disk within meshes/[mesh_name]_curvature folder in zarr structure. Filename
            corresponds to object label id.
        save_hulls: if True, save the calculated convex hull and bounding box as .vtp meshes within
            meshes/[mesh_name]_convex_hull and meshes/[mesh_name]_bounding_box directories

    """
    logger.info(
        f"Running for {zarr_url=}. \n"
        f"Extracting features for meshes in {mesh_name=} with ROI table "
        f"{roi_table=}."
    )

    # Read ROIs of objects
    adata = ad.read_zarr(f"{zarr_url}/tables/{roi_table}")
    labels = adata.obs_vector('label')

    if len(labels) == 0:
        logger.warning("Well contains no objects")

    df_feats = []

    # TODO with NGIO refactor, consider not looping over ROI table but directly over mesh names. This will
    #  generalize task to not require a corresponding ROI table in case meshes were generated outside of Fractal
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
            polydata = read_stl_polydata(mesh_path)
        else:
            logger.warning(f"No mesh found for label {org_label}")
            continue

        ##############
        # Extract features  ###
        ##############
        measurements = {'label': org_label}
        vtk_measurements, convex_hull_polydata, bounding_box_polydata = get_mesh_measurements(polydata)

        ##############
        # Combine measurements and save  ###
        ##############
        measurements.update(vtk_measurements)
        df_feats.append(measurements)

        if save_hulls:
            # save convex hull
            save_transform_path = f"{zarr_url}/meshes/{mesh_name}_convex_hull"
            os.makedirs(save_transform_path, exist_ok=True)
            # Save name is the organoid label id
            save_name = f"{int(org_label)}.vtp"
            export_vtk_polydata(os.path.join(save_transform_path, save_name), convex_hull_polydata)

            # save bounding box
            save_transform_path = f"{zarr_url}/meshes/{mesh_name}_bounding_box"
            os.makedirs(save_transform_path, exist_ok=True)
            export_vtk_polydata(os.path.join(save_transform_path, save_name), bounding_box_polydata)

        if calculate_curvature:
            # Calculate curvature
            polydata_curv, scalar_range, curvatures_numpy = get_gaussian_curvatures(polydata)
            # Save mesh
            save_transform_path = f"{zarr_url}/meshes/{mesh_name}_curvature"
            os.makedirs(save_transform_path, exist_ok=True)
            # Save name is the organoid label id
            save_name_curv = f"{int(org_label)}.vtp"
            export_vtk_polydata(os.path.join(save_transform_path, save_name_curv), polydata_curv)

        logger.info(f"Successfully extracted features for mesh {mesh_fname}.")

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
    logger.info(f"Successfully saved extracted features as feature table {output_table_name}")
    logger.info(f"End scmultiplex_mesh_measurements task for {zarr_url=}")

    return {}


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=scmultiplex_mesh_measurements,
        logger_name=logger.name,
    )
