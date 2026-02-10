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
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fractal_tasks_core.tasks.io_models import InitArgsRegistrationConsensus
from ngio import open_ome_zarr_container
from ngio.utils import ngio_logger
from pydantic import validate_call
from scipy.spatial import cKDTree
from vtkmodules.util import numpy_support

from scmultiplex.fractal.fractal_helper_functions import extract_acq_info
from scmultiplex.meshing.MeshFunctions import export_vtk_polydata, load_mesh_as_polydata
from scmultiplex.meshing.MeshProjection import assign_vertices_to_nuclei

logger = logging.getLogger(__name__)
ngio_logger.setLevel("ERROR")


@validate_call
def annotate_mesh_by_child_features(
    *,
    # Fractal arguments
    zarr_url: str,
    init_args: InitArgsRegistrationConsensus,
    # Task-specific arguments
    annotate_mesh_points_by_closest_cell: bool = False,
    annotate_mesh_points_by_all_projected_cells: bool = False,
    parent_mesh_name: str,
    parent_roi_table_name: str,
    child_feature_table_name: str,
    annotate_by_features: list[str],
    parent_of_child_colname: str = "ROI_label",
    new_mesh_name: str,
    save_nonsurface_labels: bool = True,
    maximum_distance: float = 20.0,
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
        annotate_mesh_points_by_closest_cell: If True, each mesh point is assigned to the closest child label centroid
            by Euclidean distance (L2 norm). This prioritizes cells closer to mesh surface and thus may miss cells
            deeper in tissue.
        annotate_mesh_points_by_all_projected_cells: If True, all cell centroids are projected to mesh
            surface and expanded via Voronoi tesselation. This ensures no cells are missed, but simplifies stratefied
            epithelium to a 2D plane. While cell boundaries are not accurate, neighborhood relations are preserved.
        parent_mesh_name: Mesh folder name on which annotation is performed. Must contain .stl format meshes whose
            filename corresponds to label name in ROI table.
        parent_roi_table_name: Name of the ROI table that corresponds to labels of meshed objects, only used for indexing
            objects in for loop.
        child_feature_table_name: Name of the feature table extracted from child objects. Assumes that it contains columns
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
        save_nonsurface_labels: If True, saves a .npy for each organoid id with list of label ids to remove.
        maximum_distance: Only used if annotate_mesh_points_by_all_projected_cells is True. Sets distance cutoff
            in same units as mesh (typically physical units, um) above which a nucleus is not mapped to mesh surface.
            This prevents debris from object center to be mapped to surface. For example, if set to 20, cell centroids
            that are more than 20 away from closest mesh point are not mapped to surface.

    """
    logger.info(
        f"Running for {zarr_url=}. \n"
        f"Annotating meshes in {parent_mesh_name=} with features {annotate_by_features} from feature table "
        f"{child_feature_table_name=}."
    )

    if (
        annotate_mesh_points_by_closest_cell
        == annotate_mesh_points_by_all_projected_cells
    ):
        raise ValueError(
            "Exactly one of "
            "'annotate_mesh_points_by_closest_cell' or "
            "'annotate_mesh_points_by_all_projected_cells' must be True."
        )

    # Zarr_url is the url of the reference round
    ome_zarr = open_ome_zarr_container(zarr_url)
    ref_round_id = extract_acq_info(zarr_url)

    # Load ROI table
    parent_roi_table = ome_zarr.get_table(
        parent_roi_table_name, check_type="generic_roi_table"
    )

    # Set input mesh directory
    mesh_dir = Path(zarr_url) / "meshes" / parent_mesh_name

    logger.info(
        f"Reading features from round(s): {[extract_acq_info(url) for url in init_args.zarr_url_list]}"
    )
    # Key is round ID, value is pandas df of child features (for all parent objects)
    features_dict = {}
    # Load child features for all submitted rounds
    for acq_zarr_url in init_args.zarr_url_list:
        round_id = extract_acq_info(acq_zarr_url)

        c = open_ome_zarr_container(acq_zarr_url)
        df = c.get_table(child_feature_table_name, check_type="feature_table").dataframe

        features_dict[round_id] = df

    # Get dtype of [parent_of_child_colname] in reference child features
    reference_feat_df = features_dict[ref_round_id]
    col_dtype = reference_feat_df[parent_of_child_colname].dtype

    convert_label_to_numeric = pd.api.types.is_numeric_dtype(col_dtype)

    if convert_label_to_numeric:
        logger.info(f"Detected numeric datatype in column {parent_of_child_colname=}.")

    # Initializing saving location
    save_mesh_path = f"{zarr_url}/meshes/{new_mesh_name}"
    os.makedirs(save_mesh_path, exist_ok=True)
    logger.info(f"Set output save path to reference directory: {save_mesh_path}")

    # Intialize dictionary in case saving nonsurface labels
    nonsurface_dict = {}

    if annotate_mesh_points_by_closest_cell:
        logger.info("Starting assignment of mesh points to closest cell centroid.")
    elif annotate_mesh_points_by_all_projected_cells:
        logger.info("Starting projection of cell centroids to mesh points.")

    # for every object in ROI table...
    for roi in parent_roi_table.rois():
        label_string = roi.name

        if convert_label_to_numeric:
            label_value = col_dtype.type(label_string)
        else:
            label_value = label_string

        # Load parent object mesh
        polydata = load_mesh_as_polydata(mesh_dir, label_string)

        if polydata is None:
            # No mesh found for any of the supported extensions
            logger.warning(
                f"No mesh found for label {label_string}. Supported file types: [.stl, .vtk, .vtp]."
            )
            continue

        ##############
        # Assign mesh points to nuclei  ###
        ##############

        # select features table that correspond to specific parent object
        feat_sel_df = reference_feat_df[
            reference_feat_df[parent_of_child_colname] == label_value
        ]

        if feat_sel_df.empty:
            logger.warning(
                f"No child objects found in object {label_string}. Skipping."
            )
            continue

        # numpy (N_cells, 3)
        # physical units (um)
        feat_xyz = feat_sel_df[
            ["x_pos_pix", "y_pos_pix", "z_pos_pix_scaled"]
        ].to_numpy()

        label_array = feat_sel_df.index.to_numpy()

        # get mesh points
        vtk_points = polydata.GetPoints()
        points_array = numpy_support.vtk_to_numpy(
            vtk_points.GetData()
        )  # shape: (N_points, 3)

        mesh_df = pd.DataFrame(points_array, columns=["x", "y", "z"])

        # numpy (N_points, 3)
        mesh_xyz = mesh_df[["x", "y", "z"]].to_numpy()

        logger.info(
            f"Assigning child centroids to mesh points for object {label_string}..."
        )

        if annotate_mesh_points_by_closest_cell:
            # Build KD-tree from points in feature table
            tree = cKDTree(feat_xyz)

            # Query the nearest neighbor in feature table for each point in mesh
            # length of indices equals number of rows in mesh_df (i.e. all mesh points)
            # indices contains row index into feat_df of the nearest feature point
            distances, indices = tree.query(mesh_xyz)

        if annotate_mesh_points_by_all_projected_cells:

            vtk_polys = polydata.GetPolys()
            # [num_points_in_face, idx0, idx1, idx2, num_points_in_face, idx3, idx4, idx5...]
            polys_array = numpy_support.vtk_to_numpy(vtk_polys.GetData())  # 1D

            if not np.all(polys_array[::4] == 3):
                raise ValueError("Non-triangular faces detected.")

            # Reshape to (-1, 4): [idx0, idx1, idx2]
            mesh_faces = polys_array.reshape(-1, 4)[
                :, 1:
            ]  # drop first column num_points_in_face

            indices, proj_vertex_ids = assign_vertices_to_nuclei(
                mesh_xyz, mesh_faces, feat_xyz, max_dist=maximum_distance
            )
        ##############
        # Annotate point data with feature values  ###
        ##############

        logger.info("Annotating mesh by features...")
        for acq_zarr_url in init_args.zarr_url_list:
            round_id = extract_acq_info(acq_zarr_url)

            # Get all features for this round from dictionary
            annot_feat_df = features_dict[round_id]

            # Select all features that correspond to specific parent object
            annot_feat_sel_df = annot_feat_df[
                annot_feat_df[parent_of_child_colname] == label_value
            ]

            annot_label_array = annot_feat_sel_df.index.to_numpy()

            if not np.array_equal(label_array, annot_label_array):
                raise ValueError(
                    f"Child label order does not match across rounds. "
                    f"\nReference labels: {label_array} "
                    f"\nRound {round_id} labels: {annot_label_array}"
                )

            # For user-specified features to color by...
            for col in annotate_by_features:
                # Define round-based column name
                save_col_name = f"{str(round_id)}.{col}"

                if col not in annot_feat_sel_df.columns:
                    # Skip column names not in data table
                    continue
                else:
                    # Get column as Numpy array
                    col_array = annot_feat_sel_df.loc[
                        annot_feat_sel_df[parent_of_child_colname] == label_value, col
                    ].to_numpy()

                if len(feat_xyz) != len(col_array):
                    raise ValueError("Unequal feature lengths between rounds")

                # Assign the corresponding feature value, add to mesh_df as column
                intensity_array = col_array[indices]
                mesh_df[col] = intensity_array
                # Convert to VTK array
                vtk_intensity = numpy_support.numpy_to_vtk(intensity_array)
                vtk_intensity.SetName(save_col_name)  # Name of the array in VTK
                polydata.GetPointData().AddArray(vtk_intensity)

                # TODO: Do not hardcode "label" here
                if (
                    save_nonsurface_labels
                    and col == "label"
                    and round_id == ref_round_id
                ):
                    # Save nonsurface labels (to remove) only for reference round and "label" column
                    surface_labels = mesh_df[col].unique()
                    all_labels = annot_feat_sel_df[col].unique()
                    nonsurface_labels = list(set(all_labels) - set(surface_labels))
                    # Assign value for this object to dict
                    nonsurface_dict[label_string] = nonsurface_labels

        ##############
        # Save mesh  ###
        ##############

        # Save name is the organoid label id
        save_name = f"{int(label_string)}.vtp"
        export_vtk_polydata(os.path.join(save_mesh_path, save_name), polydata)
        logger.info(f"Saved mesh as '{save_name}'")

        # Save nonsurface labels
        if save_nonsurface_labels and nonsurface_dict:
            save_dict = {
                str(k): np.array(v, dtype=np.int32) for k, v in nonsurface_dict.items()
            }
            # if dictionary not empty, save it in reference round
            registration_folder_path = os.path.join(
                zarr_url, "registration", "nonsurface_labels"
            )
            os.makedirs(registration_folder_path, exist_ok=True)
            filename = "nonsurface_labels_to_remove.npz"
            registration_save_path = os.path.join(registration_folder_path, filename)
            # Save to .npz file
            np.savez(registration_save_path, **save_dict)
            logger.info(f"Saved nonsurface labels in {registration_save_path}.")

    logger.info(f"End annotate_mesh_by_child_features task for {zarr_url=}")

    return {}


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=annotate_mesh_by_child_features,
        logger_name=logger.name,
    )
