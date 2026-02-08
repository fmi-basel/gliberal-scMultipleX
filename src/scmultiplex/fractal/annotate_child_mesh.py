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

import numpy as np
from anndata._core.aligned_df import ImplicitModificationWarning
from fractal_tasks_core.tasks.io_models import InitArgsRegistrationConsensus
from ngio import open_ome_zarr_container
from ngio.utils import ngio_logger
from pydantic import validate_call
from vtkmodules.util import numpy_support

from scmultiplex.fractal.fractal_helper_functions import extract_acq_info
from scmultiplex.meshing.MeshFunctions import export_vtk_polydata, load_mesh_as_polydata

warnings.filterwarnings("ignore", category=ImplicitModificationWarning)

logger = logging.getLogger(__name__)
ngio_logger.setLevel("ERROR")


@validate_call
def annotate_child_mesh(
    *,
    # Fractal arguments
    zarr_url: str,
    init_args: InitArgsRegistrationConsensus,
    # Task-specific arguments
    grouped_mesh_name: str,
    parent_roi_table_name: str,
    child_feature_table_name: str,
    annotate_by_features: list[str],
    parent_of_child_colname: str = "ROI_label",
    new_mesh_name: str,
) -> dict[str, Any]:
    """
    Annotate grouped child mesh (.vtp) vertices by child features, save as .vtp mesh.

    Only meshes in reference round are annotated, run on reference round but know moving rounds.
    Moving round is added as appendix.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            Refers to the zarr_url of the reference acquisition.
        init_args: Intialization arguments provided by
            `init_select_reference_knowing_all`. They contain the all zarr_urls submitted, both ref and moving.
        grouped_mesh_name: Mesh folder name on which annotation is performed. Must contain .vtp format meshes whose
            filename corresponds to label name in ROI table. Usually these are grouped child labels per parent object.
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
    """
    logger.info(
        f"Running for {zarr_url=}. \n"
        f"Annotating meshes in {grouped_mesh_name=} with features {annotate_by_features} from feature table "
        f"{child_feature_table_name=}."
    )

    # Zarr_url is the url of the reference round
    ome_zarr = open_ome_zarr_container(zarr_url)
    ref_round_id = extract_acq_info(zarr_url)

    # Load ROI table
    parent_roi_table = ome_zarr.get_table(
        parent_roi_table_name, check_type="generic_roi_table"
    )

    # Set input mesh directory
    mesh_dir = f"{zarr_url}/meshes/{grouped_mesh_name}"

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

    # Initializing saving location
    save_mesh_path = f"{zarr_url}/meshes/{new_mesh_name}"
    os.makedirs(save_mesh_path, exist_ok=True)

    logger.info(f"Set output save path to reference directory: {save_mesh_path}")

    logger.info("Starting annotation of child mesh by features.")

    # for every object in ROI table...
    for roi in parent_roi_table.rois():
        label_string = roi.name
        label_dtyped = np.dtype(col_dtype).type(label_string)

        # TODO: do not hard code file type
        mesh_fname = label_string + ".vtp"

        logger.info(f"Loading grouped mesh {mesh_fname}...")

        # Load parent object mesh
        polydata = load_mesh_as_polydata(mesh_dir, label_string)

        if polydata is None:
            # No mesh found for any of the supported extensions
            logger.warning(
                f"No mesh found for label {label_string}. Supported file types: [.stl, .vtk, .vtp]."
            )
            continue

        ##############
        # Annotate point data with feature values  ###
        ##############
        # get mesh points
        point_data = polydata.GetPointData()

        # Get label_id array (assumed to be per-point)
        label_id_vtk = point_data.GetArray("label_id")

        if label_id_vtk is None:
            raise ValueError("No 'label_id' array found in point data.")

        label_ids = numpy_support.vtk_to_numpy(label_id_vtk)  # shape: (n_points,)
        unique_label_ids = np.unique(label_ids)

        logger.info(
            f"Annotating {len(unique_label_ids)} unique labels in {mesh_fname}..."
        )

        # get feature table data
        # loop over multiple moving rounds, add to polydata
        for acq_zarr_url in init_args.zarr_url_list:

            round_id = extract_acq_info(acq_zarr_url)

            # Get all features for this round from dictionary
            annot_feat_df = features_dict[round_id]

            # Select all features that correspond to specific parent object
            annot_feat_sel_df = annot_feat_df[
                annot_feat_df[parent_of_child_colname] == label_dtyped
            ]

            if annot_feat_sel_df.empty:
                logger.warning(
                    f"No child objects found in object {label_string}. Skipping."
                )
                continue

            for col in annotate_by_features:

                if col not in annot_feat_sel_df.columns:
                    # Skip column names not in data table
                    continue

                # Define round-based column name
                save_col_name = f"{str(round_id)}.{col}"

                # Assign the corresponding feature value

                # Create array for all points
                # match dtype of original column
                new_array = np.zeros_like(label_ids, dtype=annot_feat_sel_df[col].dtype)

                # Map label_id -> value
                # key is label id, value is specific feature value for that label
                feat_map = annot_feat_sel_df[col].to_dict()  # 'label' is the index now

                for id in unique_label_ids:
                    if id in feat_map:
                        # check that label_ids and id have same dtype
                        new_array[label_ids == id] = feat_map[id]
                    else:
                        # if label does not exist set value to NA
                        new_array[label_ids == id] = np.nan

                # Convert to VTK array
                vtk_array = numpy_support.numpy_to_vtk(new_array, deep=True)
                vtk_array.SetName(save_col_name)
                polydata.GetPointData().AddArray(vtk_array)

        ##############
        # Save mesh  ###
        ##############

        # Save name is the organoid label id
        save_name = f"{int(label_string)}.vtp"
        export_vtk_polydata(os.path.join(save_mesh_path, save_name), polydata)

        logger.info(f"Saved annotated mesh as {mesh_fname}.")

    logger.info(f"End annotate_child_mesh task for {zarr_url=}")

    return {}


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=annotate_child_mesh,
        logger_name=logger.name,
    )
