import warnings
from pathlib import Path

import anndata as ad
import pandas as pd
import pytest
from fractal_tasks_core.channels import ChannelInputModel
from fractal_tasks_core.zarr_utils import OverwriteNotAllowedError
from pandas.testing import assert_frame_equal

from scmultiplex.fractal.scmultiplex_feature_measurements import (
    scmultiplex_feature_measurements,
)

multi_input_channels = {
    "C01": ChannelInputModel(wavelength_id="A01_C01"),
    "C02": ChannelInputModel(wavelength_id="A01_C01"),
    "C03": ChannelInputModel(wavelength_id="A01_C01"),
}
single_input_channels = {"C01": ChannelInputModel(wavelength_id="A01_C01")}

# TODO: Test measurements for level != 0? Not well supported yet
level = 0
label_level = 0

component_2D = "20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr/B/03/0"
component_3D = "20200812-CardiomyocyteDifferentiation14-Cycle1.zarr/B/03/0"


def load_features_for_well(table_path):
    with warnings.catch_warnings():
        adata = ad.read_zarr(table_path)
    df = adata.to_df()
    df_labels = adata.obs
    df_labels["index"] = df_labels.index
    df["index"] = df.index
    df = pd.merge(df_labels, df, on="index")
    return df


# Inputs: input_ROI_table, measure_morphology, allow_duplicate_labels, expected to run
inputs_2D = [
    ("well_ROI_table", multi_input_channels, False, False, True),
    ("well_ROI_table", None, True, False, True),
    ("well_ROI_table", {}, True, False, True),
    ("well_ROI_table", multi_input_channels, True, False, True),
    ("FOV_ROI_table", single_input_channels, False, False, False),
    ("FOV_ROI_table", single_input_channels, False, True, True),
    ("FOV_ROI_table", single_input_channels, True, True, True),
    ("well_ROI_table", None, False, False, False),  # Testing no channels, no labels
]


@pytest.mark.filterwarnings("ignore:Transforming to str index.")
@pytest.mark.parametrize(
    "input_ROI_table,input_channels,measure_morphology,allow_duplicate_labels,expected_to_run,",
    inputs_2D,
)
def test_2D_fractal_measurements(
    tiny_zenodo_zarrs_base_path,
    metadata_tiny_zenodo,
    column_names,
    input_ROI_table,
    input_channels,
    measure_morphology,
    allow_duplicate_labels,
    expected_to_run,
):
    base_path = tiny_zenodo_zarrs_base_path
    component = component_2D
    try:
        output_table_name = f"table_{input_ROI_table}_{len(input_channels)}_{measure_morphology}_{level}_{label_level}"
    except TypeError:
        output_table_name = (
            f"table_{input_ROI_table}_0_{measure_morphology}_{level}_{label_level}"
        )

    # Prepare fractal task
    label_image = "nuclei"
    if not expected_to_run:
        with pytest.raises(ValueError):
            scmultiplex_feature_measurements(
                input_paths=[base_path],
                output_path=base_path,
                metadata=metadata_tiny_zenodo["metadata_2D"],
                component=component,
                input_ROI_table=input_ROI_table,
                input_channels=input_channels,
                label_image=label_image,
                label_level=label_level,
                level=level,
                output_table_name=output_table_name,
                measure_morphology=measure_morphology,
                allow_duplicate_labels=allow_duplicate_labels,
            )
    else:
        scmultiplex_feature_measurements(
            input_paths=[base_path],
            output_path=base_path,
            metadata=metadata_tiny_zenodo["metadata_2D"],
            component=component,
            input_ROI_table=input_ROI_table,
            input_channels=input_channels,
            label_image=label_image,
            label_level=label_level,
            level=level,
            output_table_name=output_table_name,
            measure_morphology=measure_morphology,
            allow_duplicate_labels=allow_duplicate_labels,
        )

        # Check & verify the output_table
        ad_path = Path(base_path) / component / "tables" / output_table_name
        df = load_features_for_well(ad_path)

        if input_ROI_table == "well_ROI_table":
            assert len(df) == 1493
        elif input_ROI_table == "FOV_ROI_table":
            assert len(df) == 1504

        expected_columns = []
        if measure_morphology:
            expected_columns = (
                column_names["columns_2D_common"].copy()
                + column_names["columns_2D_morphology"].copy()
            )
        else:
            expected_columns = column_names["columns_2D_common"].copy()
        # Add intensity columns
        if input_channels:
            for channel in input_channels.keys():
                for feature in column_names["columns_2D_intensity"]:
                    expected_columns.append(feature.format(Ch=channel))

        assert list(df.columns) == expected_columns

        # Load expected table & compare
        expected_table_path = (
            Path(base_path) / component / "tables" / f"expected_{output_table_name}"
        )
        df_expected = load_features_for_well(expected_table_path)
        assert_frame_equal(df, df_expected)


# Inputs: input_ROI_table, measure_morphology, allow_duplicate_labels, expected to run
inputs_3D = [
    ("well_ROI_table", multi_input_channels, False, False, True),
    ("well_ROI_table", multi_input_channels, True, False, True),
    ("FOV_ROI_table", single_input_channels, False, True, True),
]


@pytest.mark.filterwarnings("ignore:Transforming to str index.")
@pytest.mark.filterwarnings(
    "ignore:Failed to get convex hull image. Returning empty image, see error message below:"
)
@pytest.mark.filterwarnings("ignore:divide by zero encountered in double_scalars")
@pytest.mark.parametrize(
    "input_ROI_table,input_channels,measure_morphology,allow_duplicate_labels,expected_to_run,",
    inputs_3D,
)
def test_3D_fractal_measurements(
    tiny_zenodo_zarrs_base_path,
    metadata_tiny_zenodo,
    column_names,
    input_ROI_table,
    input_channels,
    measure_morphology,
    allow_duplicate_labels,
    expected_to_run,
):
    base_path = tiny_zenodo_zarrs_base_path
    component = component_3D
    output_table_name = (
        f"table_{input_ROI_table}_{measure_morphology}_{level}_{label_level}"
    )

    # Prepare fractal task
    label_image = "nuclei"
    if not expected_to_run:
        with pytest.raises(ValueError):
            scmultiplex_feature_measurements(
                input_paths=[base_path],
                output_path=base_path,
                metadata=metadata_tiny_zenodo["metadata_3D"],
                component=component,
                input_ROI_table=input_ROI_table,
                input_channels=input_channels,
                label_image=label_image,
                label_level=label_level,
                level=level,
                output_table_name=output_table_name,
                measure_morphology=measure_morphology,
                allow_duplicate_labels=allow_duplicate_labels,
            )
    else:
        scmultiplex_feature_measurements(
            input_paths=[base_path],
            output_path=base_path,
            metadata=metadata_tiny_zenodo["metadata_3D"],
            component=component,
            input_ROI_table=input_ROI_table,
            input_channels=input_channels,
            label_image=label_image,
            label_level=label_level,
            level=level,
            output_table_name=output_table_name,
            measure_morphology=measure_morphology,
            allow_duplicate_labels=allow_duplicate_labels,
        )

        # Check & verify the output_table
        ad_path = Path(base_path) / component / "tables" / output_table_name
        df = load_features_for_well(ad_path)

        if input_ROI_table == "well_ROI_table":
            assert len(df) == 1632
        elif input_ROI_table == "FOV_ROI_table":
            assert len(df) == 1632

        if measure_morphology:
            expected_columns = (
                column_names["columns_3D_common"].copy()
                + column_names["columns_3D_morphology"].copy()
            )
        else:
            expected_columns = column_names["columns_3D_common"].copy()
        # Add intensity columns
        for channel in input_channels.keys():
            for feature in column_names["columns_3D_intensity"]:
                expected_columns.append(feature.format(Ch=channel))
        assert list(df.columns) == expected_columns

        # Load expected table & compare
        expected_table_path = (
            Path(base_path) / component / "tables" / f"expected_{output_table_name}"
        )
        df_expected = load_features_for_well(expected_table_path)
        assert_frame_equal(df, df_expected)


inputs_masked = [{}, single_input_channels]


@pytest.mark.filterwarnings("ignore:Transforming to str index.")
@pytest.mark.parametrize("input_channels,", inputs_masked)
def test_masked_measurements(
    tiny_zenodo_zarrs_base_path,
    metadata_tiny_zenodo,
    column_names,
    input_channels,
):
    # Test measuring when using a ROI table with masks
    allow_duplicate_labels = False
    base_path = tiny_zenodo_zarrs_base_path
    component = component_2D
    input_ROI_table = "nuclei_ROI_table"
    measure_morphology = True
    output_table_name = f"table_masked_{input_ROI_table}_{len(input_channels)}_{measure_morphology}_{level}_{label_level}"

    # Prepare fractal task
    label_image = "nuclei"

    scmultiplex_feature_measurements(
        input_paths=[base_path],
        output_path=base_path,
        metadata=metadata_tiny_zenodo["metadata_2D"],
        component=component,
        input_ROI_table=input_ROI_table,
        input_channels=input_channels,
        label_image=label_image,
        label_level=label_level,
        level=level,
        output_table_name=output_table_name,
        measure_morphology=measure_morphology,
        allow_duplicate_labels=allow_duplicate_labels,
    )

    # Check & verify the output_table
    ad_path = Path(base_path) / component / "tables" / output_table_name
    df = load_features_for_well(ad_path)

    assert len(df) == 1493

    expected_columns = []
    if measure_morphology:
        expected_columns = (
            column_names["columns_2D_common"].copy()
            + column_names["columns_2D_morphology"].copy()
        )
    else:
        expected_columns = column_names["columns_2D_common"].copy()
    # Insert ROI_label entry to columns
    expected_columns.insert(3, "ROI_label")
    # Add intensity columns
    for channel in input_channels.keys():
        for feature in column_names["columns_2D_intensity"]:
            expected_columns.append(feature.format(Ch=channel))

    assert list(df.columns) == expected_columns

    # Load expected table & compare
    expected_table_path = (
        Path(base_path) / component / "tables" / f"expected_{output_table_name}"
    )
    df_expected = load_features_for_well(expected_table_path)
    assert_frame_equal(df, df_expected)


inputs_empty = [
    ({}, True),
    ({}, False),
    (single_input_channels, True),
    (single_input_channels, False),
]


@pytest.mark.parametrize("input_channels,measure_morphology", inputs_empty)
def test_empty_label(
    tiny_zenodo_zarrs_base_path,
    metadata_tiny_zenodo,
    input_channels,
    measure_morphology,
):
    input_ROI_table = "well_ROI_table"
    base_path = tiny_zenodo_zarrs_base_path
    component = component_2D
    output_table_name = f"empty_{input_ROI_table}_{len(input_channels)}_{measure_morphology}_{level}_{label_level}"

    # Prepare fractal task
    label_image = "empty"

    scmultiplex_feature_measurements(
        input_paths=[base_path],
        output_path=base_path,
        metadata=metadata_tiny_zenodo["metadata_2D"],
        component=component,
        input_ROI_table=input_ROI_table,
        input_channels=input_channels,
        label_image=label_image,
        label_level=label_level,
        level=level,
        output_table_name=output_table_name,
        measure_morphology=measure_morphology,
    )

    # Check & verify the output_table
    ad_path = Path(base_path) / component / "tables" / output_table_name
    adata = ad.read_zarr(ad_path)
    assert len(adata) == 0
    assert adata.shape == (0, 0)


@pytest.mark.filterwarnings("ignore:Transforming to str index.")
@pytest.mark.filterwarnings(
    "ignore:The dtype argument will be deprecated in anndata 0.10.0"
)
@pytest.mark.parametrize("overwrite", [True, False])
def test_overwrite(
    tiny_zenodo_zarrs_base_path,
    metadata_tiny_zenodo,
    overwrite: bool,
):
    input_ROI_table = "well_ROI_table"
    input_channels = multi_input_channels
    measure_morphology = False
    allow_duplicate_labels = False
    base_path = tiny_zenodo_zarrs_base_path
    component = component_2D
    try:
        output_table_name = f"table_overwrite_{overwrite}"
    except TypeError:
        output_table_name = (
            f"table_{input_ROI_table}_0_{measure_morphology}_{level}_{label_level}"
        )

    # Prepare fractal task
    label_image = "nuclei"
    scmultiplex_feature_measurements(
        input_paths=[base_path],
        output_path=base_path,
        metadata=metadata_tiny_zenodo["metadata_2D"],
        component=component,
        input_ROI_table=input_ROI_table,
        input_channels=input_channels,
        label_image=label_image,
        label_level=label_level,
        level=level,
        output_table_name=output_table_name,
        measure_morphology=measure_morphology,
        allow_duplicate_labels=allow_duplicate_labels,
        overwrite=True,
    )

    if overwrite:
        scmultiplex_feature_measurements(
            input_paths=[base_path],
            output_path=base_path,
            metadata=metadata_tiny_zenodo["metadata_2D"],
            component=component,
            input_ROI_table=input_ROI_table,
            input_channels=input_channels,
            label_image=label_image,
            label_level=label_level,
            level=level,
            output_table_name=output_table_name,
            measure_morphology=measure_morphology,
            allow_duplicate_labels=allow_duplicate_labels,
            overwrite=overwrite,
        )

    else:
        with pytest.raises(OverwriteNotAllowedError):
            scmultiplex_feature_measurements(
                input_paths=[base_path],
                output_path=base_path,
                metadata=metadata_tiny_zenodo["metadata_2D"],
                component=component,
                input_ROI_table=input_ROI_table,
                input_channels=input_channels,
                label_image=label_image,
                label_level=label_level,
                level=level,
                output_table_name=output_table_name,
                measure_morphology=measure_morphology,
                allow_duplicate_labels=allow_duplicate_labels,
                overwrite=overwrite,
            )


# # The error I want to be raised isn't defined yet
# def test_label_image_does_not_exist():
#     input_ROI_table = "well_ROI_table"
#     input_channels = multi_input_channels
#     measure_morphology = False
#     allow_duplicate_labels = False
#     overwrite=True
#     component = component_2D
#     output_table_name = "Test"
#     # Clear prior runs
#     clear_tables_prior_run(output_table_name, component=component)

#     # Prepare fractal task
#     label_image = "nuclei_wrong"
#     with pytest.raises(NotYetDefinedError):
#         scmultiplex_feature_measurements(
#             input_paths=input_paths,
#             output_path=input_paths[0],
#             metadata=metadata_2D,
#             component=component,
#             input_ROI_table=input_ROI_table,
#             input_channels=input_channels,
#             label_image=label_image,
#             label_level=label_level,
#             level=level,
#             output_table_name=output_table_name,
#             measure_morphology=measure_morphology,
#             allow_duplicate_labels=allow_duplicate_labels,
#             overwrite=overwrite,
#         )
