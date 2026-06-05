import ngio
import pytest
from fractal_tasks_core.channels import ChannelInputModel
from ngio.utils import NgioValueError
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

image_path_2D = "20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr/B/03/0"
image_path_3D = "20200812-CardiomyocyteDifferentiation14-Cycle1.zarr/B/03/0"


# Inputs: input_ROI_table, measure_morphology, expected to run
inputs_2D = [
    ("well_ROI_table", multi_input_channels, False, True),
    ("well_ROI_table", None, True, True),
    ("well_ROI_table", {}, True, True),
    ("well_ROI_table", multi_input_channels, True, True),
    ("FOV_ROI_table", single_input_channels, False, False),
    ("well_ROI_table", None, False, False),  # Testing no channels, no labels
]


@pytest.mark.filterwarnings("ignore:Transforming to str index.")
@pytest.mark.parametrize(
    "input_ROI_table,input_channels,measure_morphology,expected_to_run,",
    inputs_2D,
)
def test_2D_fractal_measurements(
    tiny_zenodo_zarrs_base_path,
    column_names,
    input_ROI_table,
    input_channels,
    measure_morphology,
    expected_to_run,
):
    zarr_url = f"{tiny_zenodo_zarrs_base_path}/{image_path_2D}"
    try:
        output_table_name = (
            f"table_{input_ROI_table}_{len(input_channels)}_{measure_morphology}_0_0"
        )
    except TypeError:
        output_table_name = f"table_{input_ROI_table}_0_{measure_morphology}_0_0"

    # Prepare fractal task
    label_image = "nuclei"
    if not expected_to_run:
        with pytest.raises(ValueError):
            scmultiplex_feature_measurements(
                zarr_url=zarr_url,
                input_roi_table_name=input_ROI_table,
                input_channels=input_channels,
                label_name=label_image,
                output_table_name=output_table_name,
                measure_morphology=measure_morphology,
            )
    else:
        scmultiplex_feature_measurements(
            zarr_url=zarr_url,
            input_roi_table_name=input_ROI_table,
            input_channels=input_channels,
            label_name=label_image,
            output_table_name=output_table_name,
            measure_morphology=measure_morphology,
        )

        # Check & verify the output_table
        ome_zarr = ngio.open_ome_zarr_container(zarr_url)
        df = ome_zarr.get_table(output_table_name).dataframe

        if input_ROI_table == "well_ROI_table":
            assert len(df) == 1493
        elif input_ROI_table == "FOV_ROI_table":
            assert len(df) == 1504

        expected_columns = []
        meta_colums = ["ROI_table_name", "ROI_name"]
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

        expected_columns = expected_columns + meta_colums

        assert list(df.columns) == expected_columns

        # Load expected table & compare
        df_expected = ome_zarr.get_table(f"expected_{output_table_name}").dataframe
        assert_frame_equal(df, df_expected)


# Inputs: input_ROI_table, measure_morphology, expected to run
inputs_3D = [
    ("well_ROI_table", multi_input_channels, False, True),
    ("well_ROI_table", multi_input_channels, True, True),
    ("FOV_ROI_table", single_input_channels, False, True),
]


@pytest.mark.filterwarnings("ignore:Transforming to str index.")
@pytest.mark.filterwarnings(
    "ignore:Failed to get convex hull image. Returning empty image, see error message below:"
)
@pytest.mark.filterwarnings("ignore:divide by zero encountered in double_scalars")
@pytest.mark.parametrize(
    "input_ROI_table,input_channels,measure_morphology,expected_to_run,",
    inputs_3D,
)
def test_3D_fractal_measurements(
    tiny_zenodo_zarrs_base_path,
    column_names,
    input_ROI_table,
    input_channels,
    measure_morphology,
    expected_to_run,
):
    zarr_url = f"{tiny_zenodo_zarrs_base_path}/{image_path_3D}"
    output_table_name = f"table_{input_ROI_table}_{measure_morphology}_0_0"

    # Prepare fractal task
    label_image = "nuclei"
    if not expected_to_run:
        with pytest.raises(ValueError):
            scmultiplex_feature_measurements(
                zarr_url=zarr_url,
                input_roi_table_name=input_ROI_table,
                input_channels=input_channels,
                label_name=label_image,
                output_table_name=output_table_name,
                measure_morphology=measure_morphology,
            )
    else:
        scmultiplex_feature_measurements(
            zarr_url=zarr_url,
            input_roi_table_name=input_ROI_table,
            input_channels=input_channels,
            label_name=label_image,
            output_table_name=output_table_name,
            measure_morphology=measure_morphology,
        )

        # Check & verify the output_table
        ome_zarr = ngio.open_ome_zarr_container(zarr_url)
        df = ome_zarr.get_table(output_table_name).dataframe

        if input_ROI_table == "well_ROI_table":
            assert len(df) == 1632
        elif input_ROI_table == "FOV_ROI_table":
            assert len(df) == 1632

        meta_colums = ["ROI_table_name", "ROI_name"]
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
        expected_columns = expected_columns + meta_colums
        assert list(df.columns) == expected_columns

        # Load expected table & compare
        df_expected = ome_zarr.get_table(f"expected_{output_table_name}").dataframe
        assert_frame_equal(df, df_expected)


inputs_masked = [{}, single_input_channels]
inputs_masked = [single_input_channels]


@pytest.mark.filterwarnings("ignore:Transforming to str index.")
def test_masked_measurements_with_orgs_and_nuc(
    linking_zenodo_zarrs,
):
    """
    The purpose is to test masked measurements where the mask is a different
    label image than the labels to be measured. See
    https://github.com/fmi-basel/gliberal-scMultipleX/pull/122 for details.
    """
    zarr_url = f"{linking_zenodo_zarrs[0]}/C/02/0"
    input_ROI_table = "org_ROI_table"
    measure_morphology = True
    output_table_name = "measurements_nuc_masked"
    input_channels = {"C01": ChannelInputModel(wavelength_id="A04_C01")}

    # Prepare fractal task
    label_image = "nuc"

    scmultiplex_feature_measurements(
        zarr_url=zarr_url,
        input_roi_table_name=input_ROI_table,
        input_channels=input_channels,
        label_name=label_image,
        output_table_name=output_table_name,
        measure_morphology=measure_morphology,
    )

    # Check that there are measurement for all 20 nuclei (before #122,
    # there was only 1 measurements)
    # Check & verify the output_table
    ome_zarr = ngio.open_ome_zarr_container(zarr_url)
    df = ome_zarr.get_table(output_table_name).dataframe
    assert len(df) == 20


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
    zarr_url = f"{tiny_zenodo_zarrs_base_path}/{image_path_2D}"
    output_table_name = (
        f"empty_{input_ROI_table}_{len(input_channels)}_{measure_morphology}_0_0"
    )

    # Prepare fractal task
    label_image = "empty"

    scmultiplex_feature_measurements(
        zarr_url=zarr_url,
        input_roi_table_name=input_ROI_table,
        input_channels=input_channels,
        label_name=label_image,
        output_table_name=output_table_name,
        measure_morphology=measure_morphology,
    )

    # Check & verify the output_table
    ome_zarr = ngio.open_ome_zarr_container(zarr_url)
    df = ome_zarr.get_table(output_table_name).dataframe
    assert len(df) == 0
    assert df.shape == (0, 0)


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
    zarr_url = f"{tiny_zenodo_zarrs_base_path}/{image_path_2D}"
    try:
        output_table_name = f"table_overwrite_{overwrite}"
    except TypeError:
        output_table_name = f"table_{input_ROI_table}_0_{measure_morphology}_0_0"

    # Prepare fractal task
    label_image = "nuclei"
    scmultiplex_feature_measurements(
        zarr_url=zarr_url,
        input_roi_table_name=input_ROI_table,
        input_channels=input_channels,
        label_name=label_image,
        output_table_name=output_table_name,
        measure_morphology=measure_morphology,
        overwrite=True,
    )

    if overwrite:
        scmultiplex_feature_measurements(
            zarr_url=zarr_url,
            input_roi_table_name=input_ROI_table,
            input_channels=input_channels,
            label_name=label_image,
            output_table_name=output_table_name,
            measure_morphology=measure_morphology,
            overwrite=overwrite,
        )

    else:
        with pytest.raises(NgioValueError):
            scmultiplex_feature_measurements(
                zarr_url=zarr_url,
                input_roi_table_name=input_ROI_table,
                input_channels=input_channels,
                label_name=label_image,
                output_table_name=output_table_name,
                measure_morphology=measure_morphology,
                overwrite=overwrite,
            )
