from pandas.testing import assert_frame_equal
from pathlib import Path
import pandas as pd
import anndata as ad
import os
import shutil
import json
import pytest
import warnings

from scmultiplex.fractal.scmultiplex_feature_measurements import scmultiplex_measurements

input_paths = ["resources/scMultipleX_testdata/"]
metadata_2D ={
  "plate": [
    "20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr"
  ],
  "well": [
    "20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr/B/03/"
  ],
  "image": [
    "20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr/B/03/0/"
  ],
  "num_levels": 5,
  "coarsening_xy": 2,
  "image_extension": "png",
  "image_glob_patterns": None
}

metadata_3D ={
  "plate": [
    "20200812-CardiomyocyteDifferentiation14-Cycle1.zarr"
  ],
  "well": [
    "20200812-CardiomyocyteDifferentiation14-Cycle1.zarr/B/03/"
  ],
  "image": [
    "20200812-CardiomyocyteDifferentiation14-Cycle1.zarr/B/03/0/"
  ],
  "num_levels": 5,
  "coarsening_xy": 2,
  "image_extension": "png",
  "image_glob_patterns": None
}

columns_2D_common = [
    'label', 'ROI_table_name', 'ROI_name', 'index', 'x_pos_pix', 'y_pos_pix'
]
columns_2D_morphology = [
    'is_touching_border_xy', 'imgdim_x', 'imgdim_y', 
    'area_bbox', 'area_convhull', 'equivDiam', 'extent', 'solidity', 
    'majorAxisLength', 'minorAxisLength', 'minmajAxisRatio', 
    'aspectRatio_equivalentDiameter', 'area_pix', 'perimeter', 'concavity', 
    'asymmetry', 'eccentricity', 'circularity', 'concavity_count', 
    'disconnected_components'
]
columns_2D_intensity = [
    'C01.mean_intensity', 'C01.max_intensity', 'C01.min_intensity', 
    'C01.percentile25', 'C01.percentile50', 'C01.percentile75', 
    'C01.percentile90', 'C01.percentile95', 'C01.percentile99', 'C01.stdev', 
    'C01.skew', 'C01.kurtosis', 'C01.x_pos_weighted_pix', 
    'C01.y_pos_weighted_pix', 'C01.x_massDisp_pix', 'C01.y_massDisp_pix'
]

columns_3D_common = [
    'label', 'ROI_table_name', 'ROI_name', 'index', 'x_pos_pix', 'y_pos_pix', 
    'z_pos_pix_scaled', 'z_pos_pix_img', 'volume_pix',
]
columns_3D_morphology = [
    'is_touching_border_xy', 'imgdim_x', 'imgdim_y', 'area_bbox', 
    'area_convhull', 'equivDiam', 'extent', 'solidity', 'majorAxisLength', 
    'minorAxisLength', 'minmajAxisRatio', 'aspectRatio_equivalentDiameter',
    'imgdim_z', 'is_touching_border_z', 'surface_area'
]

columns_3D_intensity = [
    'C01.mean_intensity', 'C01.max_intensity', 'C01.min_intensity', 
    'C01.percentile25', 'C01.percentile50', 'C01.percentile75', 
    'C01.percentile90', 'C01.percentile95', 'C01.percentile99', 
    'C01.stdev', 'C01.skew', 'C01.kurtosis', 'C01.x_pos_weighted_pix', 
    'C01.y_pos_weighted_pix', 'C01.x_massDisp_pix', 'C01.y_massDisp_pix', 
    'C01.z_pos_weighted_pix', 'C01.z_massDisp_pix'
]

# TODO: Test measurements for level != 0? Not well supported yet
level = 0
label_level = 0

component_2D = "20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr/B/03/0"
component_3D = "20200812-CardiomyocyteDifferentiation14-Cycle1.zarr/B/03/0"

# Inputs: input_ROI_table, measure_morphology, allow_duplicate_labels, expected to run
inputs_2D = [
    ("well_ROI_table", False, False, True),
    ("well_ROI_table", True, False, True),
    ("FOV_ROI_table", False, False, False),
    ("FOV_ROI_table", False, True, True),
    ("FOV_ROI_table", True, True, True),
]


def clear_tables_prior_run(output_table_name, component):
    table_folder = Path(input_paths[0]) / component / "tables"
    existing_table_folder =  table_folder / output_table_name
    if os.path.exists(existing_table_folder) and os.path.isdir(existing_table_folder):
        shutil.rmtree(existing_table_folder)
    zattrs_file = table_folder / ".zattrs"
    with open(zattrs_file, 'r+') as f:
        data = json.load(f)

    if output_table_name in data["tables"]:
        os.remove(zattrs_file)
        data["tables"].remove(output_table_name)
        with open(zattrs_file, 'w') as f:
            json.dump(data, f, indent=4)

def load_features_for_well(table_path):
    with warnings.catch_warnings():
        adata = ad.read_zarr(table_path)
    df = adata.to_df()
    df_labels = adata.obs
    df_labels["index"] = df_labels.index
    df["index"] = df.index
    df = pd.merge(df_labels, df, on="index")
    return df


@pytest.mark.filterwarnings("ignore:Transforming to str index.")
@pytest.mark.parametrize("input_ROI_table,measure_morphology,allow_duplicate_labels,expected_to_run,", inputs_2D)
def test_2D_fractal_measurements(
        input_ROI_table, 
        measure_morphology, 
        allow_duplicate_labels, 
        expected_to_run,
    ):
    component = component_2D
    output_table_name = f'table_{input_ROI_table}_{measure_morphology}_{level}_{label_level}'
    # Clear prior runs
    clear_tables_prior_run(output_table_name, component = component)

    # Prepare fractal task
    input_channels = {
        "C01": {"wavelength_id": "A01_C01"}, 
    }
    label_image = 'nuclei'
    if not expected_to_run:
        with pytest.raises(ValueError):
            scmultiplex_measurements(
                input_paths=input_paths,
                output_path=input_paths[0],
                metadata=metadata_2D,
                component=component,
                input_ROI_table = input_ROI_table, 
                input_channels = input_channels,
                label_image = label_image,
                label_level = label_level,
                level = level,
                output_table_name = output_table_name,
                measure_morphology = measure_morphology,
                allow_duplicate_labels = allow_duplicate_labels,
            )
    else:
        scmultiplex_measurements(
            input_paths=input_paths,
            output_path=input_paths[0],
            metadata=metadata_2D,
            component=component,
            input_ROI_table = input_ROI_table, 
            input_channels = input_channels,
            label_image = label_image,
            label_level = label_level,
            level = level,
            output_table_name = output_table_name,
            measure_morphology = measure_morphology,
            allow_duplicate_labels = allow_duplicate_labels,
        )

        # Check & verify the output_table
        ad_path = Path(input_paths[0]) / component / "tables" / output_table_name
        df = load_features_for_well(ad_path)

        if input_ROI_table == "well_ROI_table":
            assert len(df) == 1402
        elif input_ROI_table == "FOV_ROI_table":
            assert len(df) == 1414

        if measure_morphology:
            expected_columns = columns_2D_common + columns_2D_morphology + columns_2D_intensity
        else:
            expected_columns = columns_2D_common + columns_2D_intensity
        assert list(df.columns) == expected_columns

        # Load expected table & compare
        expected_table_path = Path(input_paths[0]) / component / "tables" / f"expected_{output_table_name}"
        df_expected = load_features_for_well(expected_table_path)
        assert_frame_equal(df, df_expected)


# Inputs: input_ROI_table, measure_morphology, allow_duplicate_labels, expected to run
inputs_3D = [
    ("well_ROI_table", False, False, True),
    ("well_ROI_table", True, False, True),
    ("FOV_ROI_table", False, False, False),
    ("FOV_ROI_table", False, True, True),
    ("FOV_ROI_table", True, True, True),
]
@pytest.mark.filterwarnings("ignore:Transforming to str index.")
@pytest.mark.filterwarnings("ignore:Failed to get convex hull image. Returning empty image, see error message below:")
@pytest.mark.filterwarnings("ignore:divide by zero encountered in double_scalars")
@pytest.mark.parametrize("input_ROI_table,measure_morphology,allow_duplicate_labels,expected_to_run,", inputs_3D)
def test_3D_fractal_measurements(
        input_ROI_table, 
        measure_morphology, 
        allow_duplicate_labels, 
        expected_to_run,
    ):
    # input_ROI_table = "organoid_ROI_table"
    component = component_3D
    output_table_name = f'table_{input_ROI_table}_{measure_morphology}_{level}_{label_level}'
    # Clear prior runs
    clear_tables_prior_run(output_table_name, component = component)

    # Prepare fractal task
    input_channels = {
        "C01": {"wavelength_id": "A01_C01"}, 
    }
    label_image = 'nuclei'
    if not expected_to_run:
        with pytest.raises(ValueError):
            scmultiplex_measurements(
                input_paths=input_paths,
                output_path=input_paths[0],
                metadata=metadata_3D,
                component=component,
                input_ROI_table = input_ROI_table, 
                input_channels = input_channels,
                label_image = label_image,
                label_level = label_level,
                level = level,
                output_table_name = output_table_name,
                measure_morphology = measure_morphology,
                allow_duplicate_labels = allow_duplicate_labels,
            )
    else:
        
        scmultiplex_measurements(
            input_paths=input_paths,
            output_path=input_paths[0],
            metadata=metadata_3D,
            component=component,
            input_ROI_table = input_ROI_table, 
            input_channels = input_channels,
            label_image = label_image,
            label_level = label_level,
            level = level,
            output_table_name = output_table_name,
            measure_morphology = measure_morphology,
            allow_duplicate_labels = allow_duplicate_labels,
        )

        # Check & verify the output_table
        ad_path = Path(input_paths[0]) / component / "tables" / output_table_name
        df = load_features_for_well(ad_path)

        if input_ROI_table == "well_ROI_table":
            assert len(df) == 1513
        elif input_ROI_table == "FOV_ROI_table":
            assert len(df) == 1522

        if measure_morphology:
            expected_columns = columns_3D_common + columns_3D_morphology + columns_3D_intensity
        else:
            expected_columns = columns_3D_common + columns_3D_intensity
        assert list(df.columns) == expected_columns

        # Load expected table & compare
        expected_table_path = Path(input_paths[0]) / component / "tables" / f"expected_{output_table_name}"
        df_expected = load_features_for_well(expected_table_path)
        assert_frame_equal(df, df_expected)


#####
# Manual tests for failing task
# Test with the actual 3D data where it's been failing on
# # input_paths = ["/Users/joel/Desktop"]


# metadata_3D = {
#   "plate": [
#     "230123LC_LGR5TC_90.zarr"
#   ],
#   "well": [
#     "230123LC_LGR5TC_90.zarr/C/08/",
#     "230123LC_LGR5TC_90.zarr/E/08/"
#   ],
#   "image": [
#     "230123LC_LGR5TC_90.zarr/C/08/0/",
#     "230123LC_LGR5TC_90.zarr/E/08/0/"
#   ],
#   "num_levels": 5,
#   "coarsening_xy": 2,
#   "image_extension": "tif",
#   "image_glob_patterns": [
#     "*_[CE]08_*",
#     "*F015*",
#     "*"
#   ]
# }
# component_3D = "230123LC_LGR5TC_90.zarr/C/08/0/"

# @pytest.mark.filterwarnings("ignore:Transforming to str index.")
# @pytest.mark.parametrize("input_ROI_table,measure_morphology,allow_duplicate_labels,expected_to_run,", inputs_3D)
# def test_3D_fractal_measurements(
#         input_ROI_table, 
#         measure_morphology, 
#         allow_duplicate_labels, 
#         expected_to_run,
#     ):
#     # input_ROI_table = "organoid_ROI_table"
#     component = component_3D
#     output_table_name = f'table_{input_ROI_table}_{measure_morphology}_{level}_{label_level}'
#     # Clear prior runs
#     clear_tables_prior_run(output_table_name, component = component)

#     # Prepare fractal task
#     input_channels = {
#         "C01": {"wavelength_id": "A01_C01"}, 
#     }
#     input_channels = {
#         "C01": {"wavelength_id": "A04_C01"}, 
#         "C02": {"wavelength_id": "A03_C02"}, 
#     }
#     label_image = 'nuclei'
#     label_image = "cell"
#     if not expected_to_run:
#         with pytest.raises(ValueError):
#             scmultiplex_measurements(
#                 input_paths=input_paths,
#                 output_path=input_paths[0],
#                 metadata=metadata_3D,
#                 component=component,
#                 input_ROI_table = input_ROI_table, 
#                 input_channels = input_channels,
#                 label_image = label_image,
#                 label_level = label_level,
#                 level = level,
#                 output_table_name = output_table_name,
#                 measure_morphology = measure_morphology,
#                 allow_duplicate_labels = allow_duplicate_labels,
#             )
#     else:
        
#         scmultiplex_measurements(
#             input_paths=input_paths,
#             output_path=input_paths[0],
#             metadata=metadata_3D,
#             component=component,
#             input_ROI_table = input_ROI_table, 
#             input_channels = input_channels,
#             label_image = label_image,
#             label_level = label_level,
#             level = level,
#             output_table_name = output_table_name,
#             measure_morphology = measure_morphology,
#             allow_duplicate_labels = allow_duplicate_labels,
#         )

#         # Check & verify the output_table
#         ad_path = Path(input_paths[0]) / component / "tables" / output_table_name
#         df = load_features_for_well(ad_path)

#         if input_ROI_table == "well_ROI_table":
#             assert len(df) == 1402
#         elif input_ROI_table == "FOV_ROI_table":
#             assert len(df) == 1414

#         if measure_morphology:
#             expected_columns = columns_2D_pos + columns_2D_morphology + columns_2D_intensity
#         else:
#             expected_columns = columns_2D_pos + columns_2D_intensity
#         assert list(df.columns) == expected_columns

#         # Load expected table & compare
#         # expected_table_path = Path(input_paths[0]) / component / "tables" / f"expected_{output_table_name}"
#         # df_expected = load_features_for_well(expected_table_path)
#         # assert_frame_equal(df, df_expected)