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

import pandas as pd
from anndata._core.aligned_df import ImplicitModificationWarning
from fractal_tasks_core.tasks.io_models import InitArgsRegistrationConsensus
from ngio import open_ome_zarr_container
from ngio.utils import ngio_logger
from pydantic import validate_call

from scmultiplex.fractal.fractal_helper_functions import extract_acq_info
from scmultiplex.meshing.MeshFunctions import (
    add_features_to_polydata,
    export_vtk_polydata,
    filter_polydata_by_label_ids,
    get_label_ids_from_polydata,
    load_mesh_as_polydata,
)
from scmultiplex.utils.io import (
    load_csv_subset_by_well_timepoint,
    parse_timepoint_well_from_zarr_path,
)

warnings.filterwarnings("ignore", category=ImplicitModificationWarning)

logger = logging.getLogger("annotate_child_mesh")
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
    annotate_by_consolidated_csv: bool = False,
    path_to_consolidated_csv: str = None,
    filter_by_csv_columns: list[str] = None,
    colname_cell_label: str = "0.cell_label",
    colname_parent_label: str = "0.parent_label",
    colname_timepoint: str = "0.timepoint",
    colname_well: str = "0.well",
    annotate_by_feature_table: bool = False,
    parent_of_child_colname: str = "ROI_name",
    child_feature_table_name: str = None,
    annotate_by_features: list[str],
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

        grouped_mesh_name: Mesh folder name containing grouped child meshes.
            Each mesh corresponds to a parent object and contains one or more
            child objects stored as a single .vtp file. Child object identity
            must be stored in the point-data array "label_id".

        parent_roi_table_name: ROI table whose ROI names correspond to the
            grouped mesh filenames. Used to iterate over parent objects.

        annotate_by_consolidated_csv: If True, annotate meshes using a
            consolidated CSV file specified by path_to_consolidated_csv.
            Exactly one of annotate_by_consolidated_csv or
            annotate_by_feature_table must be True.

        path_to_consolidated_csv: Path to a consolidated CSV containing
            child-level features for all wells and timepoints. The CSV is
            filtered internally to the current well and timepoint.

        filter_by_csv_columns: Optional list of boolean columns in the
            consolidated CSV used to filter child objects before annotation.
            Only child objects with True in all specified columns are kept.
            Child objects failing the filter are removed from the output mesh.

        colname_cell_label: Column name in the consolidated CSV containing
            child object label IDs. These labels must correspond to the
            mesh point-data array "label_id".

        colname_parent_label: Column name in the consolidated CSV containing
            parent object label IDs.

        colname_timepoint: Column name in the consolidated CSV containing
            timepoint labels.

        colname_well: Column name in the consolidated CSV containing
            well identifiers.

        annotate_by_feature_table: If True, annotate meshes using feature
            tables stored in OME-Zarr. Exactly one of
            annotate_by_consolidated_csv or annotate_by_feature_table
            must be True.

        parent_of_child_colname: Column in the feature table containing
            the parent object label associated with each child object.
            Used to select child objects belonging to the current parent.

        child_feature_table_name: Name of the OME-Zarr feature table
            containing child-level measurements.

        annotate_by_features: List of feature names to transfer from the
            feature table or consolidated CSV to the mesh. Each feature is
            added as a point-data array on the output mesh.

        new_mesh_name: Name of the output mesh directory where annotated
            .vtp meshes will be written.
    """
    logger.info(
        f"Running for {zarr_url=}. \n"
        f"Annotating meshes in {grouped_mesh_name=} with features {annotate_by_features}"
    )

    if annotate_by_consolidated_csv and annotate_by_feature_table:
        raise ValueError(
            "Cannot annotate by both CSV and feature table; "
            "set only 'Annotate by Consolidated CSV' or 'Annotate by Feature Table' to True!"
        )

    if not annotate_by_consolidated_csv and not annotate_by_feature_table:
        raise ValueError(
            "Either 'Annotate by Consolidated CSV' or 'Annotate by Feature Table' must be set to True!"
        )

    if annotate_by_feature_table:
        logger.info(
            f"Annotating child meshes by feature table {child_feature_table_name=}."
        )
        logger.info(
            f"Reading features from round(s): {[extract_acq_info(url) for url in init_args.zarr_url_list]}"
        )

    if annotate_by_consolidated_csv:
        logger.info(
            f"Annotating child meshes by consolidated CSV located in: {path_to_consolidated_csv}."
        )
        if filter_by_csv_columns is not None:
            logger.info(
                f"Filtering child labels by .csv columns: {filter_by_csv_columns}."
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

    # Initializing saving location
    save_mesh_path = f"{zarr_url}/meshes/{new_mesh_name}"
    os.makedirs(save_mesh_path, exist_ok=True)

    logger.info(
        f"New annotated meshes will be saved to reference directory: {save_mesh_path}"
    )

    logger.info("Starting annotation of child mesh.")

    # Load features across all submitted rounds
    if annotate_by_feature_table:

        # Key is round ID, value is pandas df of child features (for all parent objects)
        features_dict = {}

        # Load child features for all submitted rounds
        for acq_zarr_url in init_args.zarr_url_list:
            round_id = extract_acq_info(acq_zarr_url)

            c = open_ome_zarr_container(acq_zarr_url)
            df = c.get_table(
                child_feature_table_name, check_type="feature_table"
            ).dataframe

            features_dict[round_id] = df

        # Get dtype of [parent_of_child_colname] in reference child features
        reference_feat_df = features_dict[ref_round_id]

        # Check if need to convert parent label from a string to numeric value
        col_dtype = reference_feat_df[parent_of_child_colname].dtype
        convert_label_to_numeric = pd.api.types.is_numeric_dtype(col_dtype)

        if convert_label_to_numeric:
            logger.info(
                f"Detected numeric datatype in column {parent_of_child_colname=}."
            )

    if annotate_by_consolidated_csv:
        # Get feature columns to get from consolidated .csv
        # Preserves order while removing duplicates
        columns_to_keep = list(
            dict.fromkeys(
                [
                    colname_parent_label,
                    colname_cell_label,
                    *(filter_by_csv_columns or []),
                    *(annotate_by_features or []),
                ]
            )
        )

        timepoint, well = parse_timepoint_well_from_zarr_path(zarr_url)

        df = load_csv_subset_by_well_timepoint(
            path=path_to_consolidated_csv,
            timepoint=timepoint,
            well=well,
            colname_timepoint=colname_timepoint,
            colname_well=colname_well,
            columns_to_keep=columns_to_keep,
        )

        # Check if need to convert parent label from a string to numeric value
        col_dtype = df[colname_parent_label].dtype
        convert_label_to_numeric = pd.api.types.is_numeric_dtype(col_dtype)

    # for every object in ROI table...
    for roi in parent_roi_table.rois():
        label_string = roi.name

        if convert_label_to_numeric:
            label_value = col_dtype.type(label_string)
        else:
            label_value = label_string

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

        if annotate_by_consolidated_csv:
            # Select features that correspond to specific parent object
            sel_df = df[df[colname_parent_label] == label_value]

            # Optional: keep only child cells where all filter columns are True
            if filter_by_csv_columns is not None:
                filter_mask = sel_df[filter_by_csv_columns].all(axis=1)
                sel_df = sel_df.loc[filter_mask]

            # set index to cell id
            sel_df = sel_df.set_index(colname_cell_label, drop=False)

            if sel_df.empty:
                logger.warning(
                    f"No child objects found in object {label_string}. Skipping."
                )
                continue

            # Filter polydata object by labels to keep
            polydata = filter_polydata_by_label_ids(
                polydata=polydata,
                keep_label_ids=sel_df.index.to_numpy(),
                label_array_name="label_id",
                logger=logger,
            )

            if polydata.GetNumberOfPoints() == 0:
                logger.warning(
                    f"No mesh points remain after filtering object {label_string}. Skipping."
                )
                continue

            # Recompute label ids after filtering polydata
            label_ids, unique_label_ids = get_label_ids_from_polydata(polydata)

            logger.info(
                f"...Annotating {len(unique_label_ids)} unique child labels in parent object {label_string}"
            )

            # Create array for all points
            add_features_to_polydata(
                polydata=polydata,
                label_ids=label_ids,
                unique_label_ids=unique_label_ids,
                feature_df=sel_df,
                annotate_by_features=annotate_by_features,
                logger=logger,
            )

        if annotate_by_feature_table:

            label_ids, unique_label_ids = get_label_ids_from_polydata(polydata)

            logger.info(
                f"...Annotating {len(unique_label_ids)} unique child labels in parent object {label_string}"
            )

            # get feature table data
            # loop over multiple moving rounds, add to polydata
            for acq_zarr_url in init_args.zarr_url_list:

                round_id = extract_acq_info(acq_zarr_url)

                # Get all features for this round from dictionary
                annot_feat_df = features_dict[round_id]

                # Select all features that correspond to specific parent object
                annot_feat_sel_df = annot_feat_df[
                    annot_feat_df[parent_of_child_colname] == label_value
                ]

                if annot_feat_sel_df.empty:
                    logger.warning(
                        f"No child objects found in object {label_string}. Skipping."
                    )
                    continue

                add_features_to_polydata(
                    polydata=polydata,
                    label_ids=label_ids,
                    unique_label_ids=unique_label_ids,
                    feature_df=annot_feat_sel_df,
                    annotate_by_features=annotate_by_features,
                    column_prefix=f"{round_id}.",
                    logger=logger,
                )

        ##############
        # Save mesh  ###
        ##############

        # Save name is the organoid label id
        save_name = f"{int(label_string)}.vtp"
        export_vtk_polydata(os.path.join(save_mesh_path, save_name), polydata)

        logger.info(f"...Saved annotated mesh as {mesh_fname}")

    logger.info(f"End annotate_child_mesh task for {zarr_url=}")

    return {}


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(task_function=annotate_child_mesh)
