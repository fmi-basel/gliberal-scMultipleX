import warnings
from pathlib import Path

import anndata as ad
import pandas as pd
import pytest
from fractal_tasks_core.channels import ChannelInputModel
from fractal_tasks_core.zarr_utils import OverwriteNotAllowedError
from pandas.testing import assert_frame_equal

from scmultiplex.fractal.calculate_linking_consensus import calculate_linking_consensus
from scmultiplex.fractal.calculate_object_linking import calculate_object_linking
from scmultiplex.fractal.relabel_by_linking_consensus import relabel_by_linking_consensus
from scmultiplex.fractal.scmultiplex_feature_measurements import (
    scmultiplex_feature_measurements,
)


channel = ChannelInputModel(wavelength_id="A04_C01")

# def load_features_for_well(table_path):
#     with warnings.catch_warnings():
#         adata = ad.read_zarr(table_path)
#     df = adata.to_df()
#     df_labels = adata.obs
#     df_labels["index"] = df_labels.index
#     df["index"] = df.index
#     df = pd.merge(df_labels, df, on="index")
#     return df


def test_2d_linking(
    linking_zenodo_zarrs_base_path,
):

    base_path = linking_zenodo_zarrs_base_path
    alignment_component = '220605_151046_mip.zarr/C/02/1'
    well_component = '220605_151046_mip.zarr/C/02'

    calculate_object_linking(
        input_paths=[base_path],
        output_path=base_path,
        component=alignment_component,
        metadata={},
        label_name='org',
        roi_table="well_ROI_table",
        reference_cycle=0,
        level=0,
        iou_cutoff=0.2,
    )

    calculate_linking_consensus(
        input_paths=[base_path],
        output_path=base_path,
        component=well_component,
        metadata={},
        roi_table="org_linking",
        reference_cycle=0,
    )

    relabel_by_linking_consensus(
        input_paths=[base_path],
        output_path=base_path,
        component=alignment_component,
        metadata={},
        label_name='org',
        roi_table="well_ROI_table",
        consensus_table="org_linking_consensus",
        table_to_relabel="org_ROI_table",
        reference_cycle=0,
    )

    # input_ROI_table = "well_ROI_table"
    # base_path = tiny_zenodo_zarrs_base_path
    # component = component_2D
    # output_table_name = f"empty_{input_ROI_table}_{len(input_channels)}_{measure_morphology}_{level}_{label_level}"
    #
    # # Prepare fractal task
    # label_image = "empty"
    #
    # scmultiplex_feature_measurements(
    #     input_paths=[base_path],
    #     output_path=base_path,
    #     metadata=metadata_tiny_zenodo["metadata_2D"],
    #     component=component,
    #     input_ROI_table=input_ROI_table,
    #     input_channels=input_channels,
    #     label_image=label_image,
    #     label_level=label_level,
    #     level=level,
    #     output_table_name=output_table_name,
    #     measure_morphology=measure_morphology,
    # )



