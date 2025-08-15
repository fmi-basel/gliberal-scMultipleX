# Copyright 2025 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Nicole Repina <nicole.repina@fmi.ch>
#

"""
Annotate parent mesh vertices by child features.
"""

import logging
import os
import warnings
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from anndata._core.aligned_df import ImplicitModificationWarning
from fractal_tasks_core.tasks.io_models import InitArgsRegistrationConsensus
from pydantic import validate_call
from scipy.spatial import cKDTree
from vtkmodules.util import numpy_support

from scmultiplex.fractal.fractal_helper_functions import extract_acq_info, get_zattrs
from scmultiplex.meshing.MeshFunctions import export_vtk_polydata, read_stl_polydata

warnings.filterwarnings("ignore", category=ImplicitModificationWarning)

logger = logging.getLogger(__name__)


@validate_call
def annotate_mesh_by_child_features(
    *,
    # Fractal arguments
    zarr_url: str,
    init_args: InitArgsRegistrationConsensus,
    # Task-specific arguments
    parent_mesh_name: str,
    parent_roi_table: str,
    child_feature_table: str,
    annotate_by_features: list[str],
    parent_of_child_colname: str = "ROI_label",
    new_mesh_name: str,
) -> dict[str, Any]:
    """
    Annotate parent mesh (.stl) vertices by child features, save as .vtp mesh.

    Parent mesh vertices (i.e. points of organoid mesh) are annotated by child (e.g. single-cell)
    features. The single-cell features are mapped to the mesh surface by minimizing the 3D euclidian distance.
    Each point in mesh is assigned the value of the closest child object centroid using KDTree. User specifies
    which features from feature table should be added to .vtp point data. Output of task is .vtp mesh
    identical to parent_mesh .stl with each point annotated with desired features, for visualization in
    Paraview or conversion to a Blender-compatible format.

    Only meshes in reference round are annotated, run on reference round but know moving rounds.
    Moving round is added as appendix.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            Refers to the zarr_url of the reference acquisition.
        init_args: Intialization arguments provided by
            `init_select_reference_knowing_all`. They contain the all zarr_urls submitted, both ref and moving.
        parent_mesh_name: Mesh folder name on which annotation is performed. Must contain .stl format meshes whose
            filename corresponds to label name in ROI table.
        parent_roi_table: Name of the ROI table that corresponds to labels of meshed objects, only used for indexing
            objects in for loop.
        child_feature_table: Name of the feature table extracted from child objects. Assumes that it contains columns
            called ['x_pos_pix', 'y_pos_pix', 'z_pos_pix_scaled'] for the x,y,z centroids of each object. Assumes that
            these centroid units and scaling matches the mesh point units and scaling.
        annotate_by_features: List of strings. Each string is a column name of the child feature table and will be added
            to the .vtp mesh as a point annotation. Input should exactly match naming in child_feature_table
            columns. instance_key of child feature (e.g. "label") is added as column internally, so can be used
            as annotation feature as well.
        parent_of_child_colname: Name of column for parent label id's of child objects, e.g. the organoid id of the
            parent organoid. This column is assumed to be within obs of the child_feature_table. Usually this is
            defined in the feature extraction task.
        new_mesh_name: Name of the new mesh folder where annotated .vtp meshes are saved.

    """
    logger.info(
        f"Running for {zarr_url=}. \n"
        f"Annotating meshes in {parent_mesh_name=} with features {annotate_by_features} from feature table "
        f"{child_feature_table=}."
    )
    # Zarr_url is the url of the reference round
    # Read ROIs of objects
    adata = ad.read_zarr(f"{zarr_url}/tables/{parent_roi_table}")
    roi_attrs = get_zattrs(f"{zarr_url}/tables/{parent_roi_table}")
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

    # Initializing saving location
    save_mesh_path = f"{zarr_url}/meshes/{new_mesh_name}"
    os.makedirs(save_mesh_path, exist_ok=True)

    logger.info(f"Saving meshes to reference directory: /meshes/{new_mesh_name}")

    # TODO with NGIO refactor, consider not looping over ROI table but directly over mesh names. This will
    #  generalize task to not require a corresponding ROI table in case meshes were generated outside of Fractal
    # for every object in ROI table...
    for i, obsname in enumerate(adata.obs_names):
        org_label = labels[i]

        if not isinstance(org_label, str):
            raise TypeError("Label index must be string. Check ROI table obs naming.")

        mesh_fname = org_label + ".stl"
        mesh_path = f"{zarr_url}/meshes/{parent_mesh_name}/{mesh_fname}"

        # Check that mesh for corresponding label id exists, if not continue to next id
        if os.path.isfile(mesh_path):
            polydata = read_stl_polydata(mesh_path)
        else:
            logger.warning(f"No mesh found for label {org_label}")
            continue

        ##############
        # Annotate point data with feature values  ###
        ##############
        # get mesh points
        vtk_points = polydata.GetPoints()
        points_array = numpy_support.vtk_to_numpy(
            vtk_points.GetData()
        )  # shape: (n_points, 3)
        mesh_df = pd.DataFrame(points_array, columns=["x", "y", "z"])

        # get feature table data
        # loop over multiple moving rounds, add to polydata
        logger.info(
            f"Annotating by round(s) {[extract_acq_info(url) for url in init_args.zarr_url_list]}..."
        )
        for acq_zarr_url in init_args.zarr_url_list:
            # Load child features
            feat_adata = ad.read_zarr(f"{acq_zarr_url}/tables/{child_feature_table}")

            round_id = extract_acq_info(acq_zarr_url)

            label_dtype = feat_adata.obs[parent_of_child_colname].dtype
            org_label_typed = np.dtype(label_dtype).type(org_label)

            # select features table that correspond to specific parent object
            mask = feat_adata.obs[parent_of_child_colname] == org_label_typed
            feat_df = pd.DataFrame(feat_adata.X[mask], columns=feat_adata.var_names)
            feat_df[instance_key] = feat_adata.obs.loc[mask, instance_key].values

            if feat_df.empty:
                logger.warning(
                    f"No child objects found in object {org_label}. Skipping."
                )
                continue

            # Build KD-tree from points in feature table
            tree = cKDTree(
                feat_df[["x_pos_pix", "y_pos_pix", "z_pos_pix_scaled"]].values
            )

            # Query the nearest neighbor in feature table for each point in mesh
            distances, indices = tree.query(mesh_df[["x", "y", "z"]].values)

            for col in annotate_by_features:
                # Define round-based column name
                save_col_name = f"{str(round_id)}.{col}"

                if col not in feat_df.columns:
                    # Skip column names not in data table
                    continue
                # Assign the corresponding feature value
                mesh_df[col] = feat_df.iloc[indices][col].values
                intensity_array = mesh_df[col].to_numpy()
                # Convert to VTK array
                vtk_intensity = numpy_support.numpy_to_vtk(intensity_array)
                vtk_intensity.SetName(save_col_name)  # Name of the array in VTK
                polydata.GetPointData().AddArray(vtk_intensity)

        ##############
        # Save mesh  ###
        ##############

        # Save name is the organoid label id
        save_name = f"{int(org_label)}.vtp"
        export_vtk_polydata(os.path.join(save_mesh_path, save_name), polydata)

        logger.info(f"Saved annotated .vtp mesh for object {org_label}.")

    logger.info(f"End annotate_mesh_by_features task for {zarr_url=}")

    return {}


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=annotate_mesh_by_child_features,
        logger_name=logger.name,
    )
