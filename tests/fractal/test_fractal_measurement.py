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

# input_paths = [Path(get_tiny_zenodo_zarr_path())]
# print(input_paths)
metadata_2D = {
    "plate": ["20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr"],
    "well": ["20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr/B/03/"],
    "image": ["20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr/B/03/0/"],
    "num_levels": 5,
    "coarsening_xy": 2,
    "image_extension": "png",
    "image_glob_patterns": None,
}

metadata_3D = {
    "plate": ["20200812-CardiomyocyteDifferentiation14-Cycle1.zarr"],
    "well": ["20200812-CardiomyocyteDifferentiation14-Cycle1.zarr/B/03/"],
    "image": ["20200812-CardiomyocyteDifferentiation14-Cycle1.zarr/B/03/0/"],
    "num_levels": 5,
    "coarsening_xy": 2,
    "image_extension": "png",
    "image_glob_patterns": None,
}

columns_2D_common = [
    "label",
    "ROI_table_name",
    "ROI_name",
    "index",
    "x_pos_pix",
    "y_pos_pix",
]
columns_2D_morphology = [
    "is_touching_border_xy",
    "imgdim_x",
    "imgdim_y",
    "area_bbox",
    "area_convhull",
    "equivDiam",
    "extent",
    "solidity",
    "majorAxisLength",
    "minorAxisLength",
    "minmajAxisRatio",
    "aspectRatio_equivalentDiameter",
    "area_pix",
    "perimeter",
    "concavity",
    "asymmetry",
    "eccentricity",
    "circularity",
    "concavity_count",
    "disconnected_components",
]
columns_2D_intensity = [
    "{Ch}.mean_intensity",
    "{Ch}.max_intensity",
    "{Ch}.min_intensity",
    "{Ch}.percentile25",
    "{Ch}.percentile50",
    "{Ch}.percentile75",
    "{Ch}.percentile90",
    "{Ch}.percentile95",
    "{Ch}.percentile99",
    "{Ch}.stdev",
    "{Ch}.skew",
    "{Ch}.kurtosis",
    "{Ch}.x_pos_weighted_pix",
    "{Ch}.y_pos_weighted_pix",
    "{Ch}.x_massDisp_pix",
    "{Ch}.y_massDisp_pix",
]

columns_3D_common = [
    "label",
    "ROI_table_name",
    "ROI_name",
    "index",
    "x_pos_pix",
    "y_pos_pix",
    "z_pos_pix_scaled",
    "z_pos_pix_img",
    "volume_pix",
]
columns_3D_morphology = [
    "is_touching_border_xy",
    "imgdim_x",
    "imgdim_y",
    "area_bbox",
    "area_convhull",
    "equivDiam",
    "extent",
    "solidity",
    "majorAxisLength",
    "minorAxisLength",
    "minmajAxisRatio",
    "aspectRatio_equivalentDiameter",
    "imgdim_z",
    "is_touching_border_z",
    "surface_area",
]

columns_3D_intensity = [
    "{Ch}.mean_intensity",
    "{Ch}.max_intensity",
    "{Ch}.min_intensity",
    "{Ch}.percentile25",
    "{Ch}.percentile50",
    "{Ch}.percentile75",
    "{Ch}.percentile90",
    "{Ch}.percentile95",
    "{Ch}.percentile99",
    "{Ch}.stdev",
    "{Ch}.skew",
    "{Ch}.kurtosis",
    "{Ch}.x_pos_weighted_pix",
    "{Ch}.y_pos_weighted_pix",
    "{Ch}.x_massDisp_pix",
    "{Ch}.y_massDisp_pix",
    "{Ch}.z_pos_weighted_pix",
    "{Ch}.z_massDisp_pix",
]

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


# def clear_tables_prior_run(output_table_name, component):
#     table_folder = Path(input_paths[0]) / component / "tables"
#     existing_table_folder = table_folder / output_table_name
#     if os.path.exists(existing_table_folder) and os.path.isdir(existing_table_folder):
#         shutil.rmtree(existing_table_folder)
#     zattrs_file = table_folder / ".zattrs"
#     with open(zattrs_file, "r+") as f:
#         data = json.load(f)

#     if output_table_name in data["tables"]:
#         os.remove(zattrs_file)
#         data["tables"].remove(output_table_name)
#         with open(zattrs_file, "w") as f:
#             json.dump(data, f, indent=4)


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
    # Clear prior runs
    # clear_tables_prior_run(output_table_name, component=component)

    # Prepare fractal task
    label_image = "nuclei"
    if not expected_to_run:
        with pytest.raises(ValueError):
            scmultiplex_feature_measurements(
                input_paths=[base_path],
                output_path=base_path,
                metadata=metadata_2D,
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
            metadata=metadata_2D,
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
            expected_columns = columns_2D_common.copy() + columns_2D_morphology.copy()
        else:
            expected_columns = columns_2D_common.copy()
        # Add intensity columns
        if input_channels:
            for channel in input_channels.keys():
                for feature in columns_2D_intensity:
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
    # Clear prior runs
    # clear_tables_prior_run(output_table_name, component=component)

    # Prepare fractal task
    label_image = "nuclei"
    if not expected_to_run:
        with pytest.raises(ValueError):
            scmultiplex_feature_measurements(
                input_paths=[base_path],
                output_path=base_path,
                metadata=metadata_3D,
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
            metadata=metadata_3D,
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
            expected_columns = columns_3D_common.copy() + columns_3D_morphology.copy()
        else:
            expected_columns = columns_3D_common.copy()
        # Add intensity columns
        for channel in input_channels.keys():
            for feature in columns_3D_intensity:
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
def test_masked_measurements(input_channels, tiny_zenodo_zarrs_base_path):
    # Test measuring when using a ROI table with masks
    allow_duplicate_labels = False
    base_path = tiny_zenodo_zarrs_base_path
    component = component_2D
    input_ROI_table = "nuclei_ROI_table"
    measure_morphology = True
    output_table_name = f"table_masked_{input_ROI_table}_{len(input_channels)}_{measure_morphology}_{level}_{label_level}"
    # Clear prior runs
    # clear_tables_prior_run(output_table_name, component=component)

    # Prepare fractal task
    label_image = "nuclei"

    scmultiplex_feature_measurements(
        input_paths=[base_path],
        output_path=base_path,
        metadata=metadata_2D,
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
        expected_columns = columns_2D_common.copy() + columns_2D_morphology.copy()
    else:
        expected_columns = columns_2D_common.copy()
    # Insert ROI_label entry to columns
    expected_columns.insert(3, "ROI_label")
    # Add intensity columns
    for channel in input_channels.keys():
        for feature in columns_2D_intensity:
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
    input_channels,
    measure_morphology,
):
    input_ROI_table = "well_ROI_table"
    base_path = tiny_zenodo_zarrs_base_path
    component = component_2D
    output_table_name = f"empty_{input_ROI_table}_{len(input_channels)}_{measure_morphology}_{level}_{label_level}"
    # Clear prior runs
    # clear_tables_prior_run(output_table_name, component=component)

    # Prepare fractal task
    label_image = "empty"

    scmultiplex_feature_measurements(
        input_paths=[base_path],
        output_path=base_path,
        metadata=metadata_2D,
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
    # Clear prior runs
    # clear_tables_prior_run(output_table_name, component=component)

    # Prepare fractal task
    label_image = "nuclei"
    scmultiplex_feature_measurements(
        input_paths=[base_path],
        output_path=base_path,
        metadata=metadata_2D,
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
            metadata=metadata_2D,
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
                metadata=metadata_2D,
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
