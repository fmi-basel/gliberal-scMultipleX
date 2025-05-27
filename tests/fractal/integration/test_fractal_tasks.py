import os
from pathlib import Path

import anndata as ad
import numpy as np
from fractal_tasks_core.channels import ChannelInputModel
from numpy.testing import assert_almost_equal

from scmultiplex.fractal._image_based_registration_hcs_allrounds_init import (
    _image_based_registration_hcs_allrounds_init,
)
from scmultiplex.fractal._image_based_registration_hcs_init import (
    _image_based_registration_hcs_init,
)
from scmultiplex.fractal._init_group_by_well_for_multiplexing import (
    _init_group_by_well_for_multiplexing,
)
from scmultiplex.fractal.calculate_linking_consensus import calculate_linking_consensus
from scmultiplex.fractal.calculate_object_linking import calculate_object_linking
from scmultiplex.fractal.calculate_platymatch_registration import (
    calculate_platymatch_registration,
)
from scmultiplex.fractal.relabel_by_linking_consensus import (
    relabel_by_linking_consensus,
)
from scmultiplex.fractal.scmultiplex_mesh_measurements import (
    scmultiplex_mesh_measurements,
)
from scmultiplex.fractal.spherical_harmonics_from_labelimage import (
    spherical_harmonics_from_labelimage,
)
from scmultiplex.fractal.surface_mesh_multiscale import surface_mesh_multiscale

name_3d = "220605_151046.zarr"
name_mip = "220605_151046_mip.zarr"

test_calculate_object_linking_expected_output = np.array(
    [[1.0, 1.0, 0.9727956], [2.0, 2.0, 0.8249731], [3.0, 3.0, 0.9644809]]
)

test_calculate_linking_consensus_expected_output = np.array(
    [[1.0, 1.0, 0.0, 1.0], [2.0, 2.0, 1.0, 2.0], [3.0, 3.0, 2.0, 3.0]]
)

test_relabel_by_linking_consensus_output_dict = {
    "0": np.array(
        [
            [13.216666, 21.016666, 0.0, 22.1, 20.583334, 0.6],
            [13.866667, 91.433334, 0.0, 18.85, 15.816667, 0.6],
            [39.0, 103.566666, 0.0, 23.183332, 28.166666, 0.6],
        ]
    ),
    "1": np.array(
        [
            [49.616665, 11.05, 0.0, 22.533333, 20.583334, 0.6],
            [51.783333, 81.9, 0.0, 18.85, 15.816667, 0.6],
            [75.4, 93.816666, 0.0, 23.4, 27.95, 0.6],
        ]
    ),
}

test_calculate_platymatch_registration_output = np.array(
    [
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
        [4.0, 4.0],
        [8.0, 7.0],
        [9.0, 8.0],
        [10.0, 9.0],
        [11.0, 10.0],
        [13.0, 12.0],
        [14.0, 13.0],
        [15.0, 14.0],
        [16.0, 15.0],
        [17.0, 16.0],
        [19.0, 17.0],
        [18.0, 18.0],
        [20.0, 19.0],
    ]
)

test_sphr_harmonics_from_labelimg_expected_output = np.array(
    [10.798882, 9.207903, 13.282207]
)

test_sphr_harmonics_from_mesh_expected_output = np.array(
    [10.046525, 8.232602, 12.735255]
)

test_scmultiplex_mesh_measurements_expected_output = np.array(
    [
        4.2192085e03,
        1.2844763e03,
        1.0172350e00,
        4.5982581e-01,
        9.9415630e-01,
        5.8436790e-03,
        7.8099235e-03,
        1.1332338e00,
        1.0085807e00,
    ]
)


def select_zarr_urls(name, linking_zenodo_zarrs):
    zarr = None
    for z in linking_zenodo_zarrs:
        if z.endswith(name):
            zarr = z
            break
    assert zarr is not None

    # construct zarr url list
    zarr_urls = [f"{zarr}/C/02/0", f"{zarr}/C/02/1"]

    return zarr_urls


def test_calculate_object_linking(linking_zenodo_zarrs, name=name_mip):
    zarr_urls = select_zarr_urls(name, linking_zenodo_zarrs)
    parallelization_list = _image_based_registration_hcs_init(
        zarr_urls=zarr_urls,
        zarr_dir="",
        reference_acquisition=0,
    )

    for img in parallelization_list["parallelization_list"]:
        zarr_url = img["zarr_url"]
        init_args = img["init_args"]
        label_name = "org"
        roi_table = "well_ROI_table"
        level = 0

        calculate_object_linking(
            zarr_url=zarr_url,
            init_args=init_args,
            label_name=label_name,
            roi_table=roi_table,
            level=level,
            iou_cutoff=0.2,
        )
        output_table_path = f"{zarr_url}/tables/{label_name}_match_table"

        output = ad.read_zarr(output_table_path).to_df().to_numpy()
        assert_almost_equal(output, test_calculate_object_linking_expected_output)


def test_calculate_linking_consensus(linking_zenodo_zarrs, name=name_mip):
    zarr_urls = select_zarr_urls(name, linking_zenodo_zarrs)
    parallelization_list = _init_group_by_well_for_multiplexing(
        zarr_urls=zarr_urls,
        zarr_dir="",
        reference_acquisition=0,
    )

    for img in parallelization_list["parallelization_list"]:
        zarr_url = img["zarr_url"]
        init_args = img["init_args"]
        roi_table = "org_match_table"

        calculate_linking_consensus(
            zarr_url=zarr_url,
            init_args=init_args,
            roi_table=roi_table,
        )

        output_table_path = f"{zarr_url}/tables/{roi_table}_consensus"

        output = ad.read_zarr(output_table_path).to_df().to_numpy()

        assert_almost_equal(output, test_calculate_linking_consensus_expected_output)


def test_relabel_by_linking_consensus(linking_zenodo_zarrs, name=name_mip):
    zarr_urls = select_zarr_urls(name, linking_zenodo_zarrs)
    parallelization_list = _image_based_registration_hcs_allrounds_init(
        zarr_urls=zarr_urls,
        zarr_dir="",
        reference_acquisition=0,
    )
    for img in parallelization_list["parallelization_list"]:
        zarr_url = img["zarr_url"]
        init_args = img["init_args"]
        label_name = "org"
        consensus_table = "org_match_table_consensus"
        table_to_relabel = "org_ROI_table"

        relabel_by_linking_consensus(
            zarr_url=zarr_url,
            init_args=init_args,
            label_name=label_name,
            consensus_table=consensus_table,
            table_to_relabel=table_to_relabel,
        )

        output_table_path = f"{zarr_url}/tables/{table_to_relabel}_linked"
        output = ad.read_zarr(output_table_path).to_df().to_numpy()
        assert_almost_equal(
            output,
            test_relabel_by_linking_consensus_output_dict[Path(zarr_url).name],
            decimal=3,
        )


def test_calculate_platymatch_registration(linking_zenodo_zarrs, name=name_3d):
    zarr_urls = select_zarr_urls(name, linking_zenodo_zarrs)
    parallelization_list = _image_based_registration_hcs_init(
        zarr_urls=zarr_urls,
        zarr_dir="",
        reference_acquisition=0,
    )

    for img in parallelization_list["parallelization_list"]:
        zarr_url = img["zarr_url"]
        init_args = img["init_args"]
        label_name_to_register = "nuc"
        label_name_obj = "org"
        roi_table = "org_ROI_table"
        channel = ChannelInputModel(wavelength_id="A04_C01")

        calculate_platymatch_registration(
            zarr_url=zarr_url,
            init_args=init_args,
            label_name_to_register=label_name_to_register,
            label_name_obj=label_name_obj,
            roi_table=roi_table,
            level=0,
            save_transformation=True,
            mask_by_parent=True,
            calculate_ffd=True,
            seg_channel=channel,
            volume_filter=True,
            volume_filter_threshold=0.10,
        )

        output_table_path_affine = (
            f"{zarr_url}/tables/{label_name_to_register}_match_table_affine"
        )
        output_affine = ad.read_zarr(output_table_path_affine).to_df().to_numpy()
        output_table_path_ffd = (
            f"{zarr_url}/tables/{label_name_to_register}_match_table_ffd"
        )
        output_ffd = ad.read_zarr(output_table_path_ffd).to_df().to_numpy()
        # test that matches are correct; ignore confidence columns
        assert_almost_equal(
            output_affine[:, 0:2],
            test_calculate_platymatch_registration_output,
            decimal=3,
        )
        assert_almost_equal(
            output_ffd[:, 0:2], test_calculate_platymatch_registration_output, decimal=3
        )


def test_surface_mesh_multiscale(linking_zenodo_zarrs, name=name_3d):
    zarr_urls = select_zarr_urls(name, linking_zenodo_zarrs)
    parallelization_list = _init_group_by_well_for_multiplexing(
        zarr_urls=zarr_urls,
        zarr_dir="",
        reference_acquisition=0,
    )
    for img in parallelization_list["parallelization_list"]:
        zarr_url = img["zarr_url"]
        init_args = img["init_args"]
        group_by = "org"
        label_name = "nuc"
        roi_table = "org_ROI_table"

        surface_mesh_multiscale(
            zarr_url=zarr_url,
            init_args=init_args,
            label_name=label_name,
            group_by=group_by,
            roi_table=roi_table,
            multiscale=True,
            save_mesh=True,
            expandby_factor=0.6,
            sigma_factor=10,
            canny_threshold=0.3,
            mask_contour_by_parent=False,
            filter_children_by_volume=True,
            child_volume_filter_threshold=0.05,
            polynomial_degree=30,
            passband=0.01,
            feature_angle=160,
            target_reduction=0.98,
            smoothing_iterations=2,
        )

        output_mesh_path = f"{zarr_url}/meshes/{group_by}_from_{label_name}"
        # check that 3 mesh files were written
        assert len(os.listdir(output_mesh_path)) == 3


def test_surface_mesh_grouped(linking_zenodo_zarrs, name=name_3d):
    zarr_urls = select_zarr_urls(name, linking_zenodo_zarrs)
    parallelization_list = _init_group_by_well_for_multiplexing(
        zarr_urls=zarr_urls,
        zarr_dir="",
        reference_acquisition=0,
    )
    for img in parallelization_list["parallelization_list"]:
        zarr_url = img["zarr_url"]
        init_args = img["init_args"]
        label_name = "nuc"
        group_by = "org"
        roi_table = "org_ROI_table"

        surface_mesh_multiscale(
            zarr_url=zarr_url,
            init_args=init_args,
            label_name=label_name,
            group_by=group_by,
            roi_table=roi_table,
            multiscale=False,
            save_mesh=True,
            expandby_factor=1.0,
            sigma_factor=6,
            canny_threshold=0.2,
            polynomial_degree=30,
            passband=0.01,
            feature_angle=160,
            target_reduction=0.97,
            smoothing_iterations=1,
        )

        output_mesh_path = f"{zarr_url}/meshes/{label_name}_grouped"
        # check that 3 mesh files were written
        assert len(os.listdir(output_mesh_path)) == 3


def test_surface_mesh_per_object(linking_zenodo_zarrs, name=name_3d):
    zarr_urls = select_zarr_urls(name, linking_zenodo_zarrs)
    parallelization_list = _init_group_by_well_for_multiplexing(
        zarr_urls=zarr_urls,
        zarr_dir="",
        reference_acquisition=0,
    )
    for img in parallelization_list["parallelization_list"]:
        zarr_url = img["zarr_url"]
        init_args = img["init_args"]
        label_name = "org"
        roi_table = "org_ROI_table"

        surface_mesh_multiscale(
            zarr_url=zarr_url,
            init_args=init_args,
            label_name=label_name,
            group_by=None,
            roi_table=roi_table,
            multiscale=False,
            save_mesh=True,
            expandby_factor=1.0,
            sigma_factor=6,
            canny_threshold=0.2,
            mask_contour_by_parent=False,
            filter_children_by_volume=False,
            child_volume_filter_threshold=0.05,
            polynomial_degree=30,
            passband=0.01,
            feature_angle=160,
            target_reduction=0.97,
            smoothing_iterations=1,
        )

        output_mesh_path = f"{zarr_url}/meshes/{label_name}"
        # check that 3 mesh files were written
        assert len(os.listdir(output_mesh_path)) == 3


def test_sphr_harmonics_from_labelimage(linking_zenodo_zarrs, name=name_3d):
    zarr_urls = select_zarr_urls(name, linking_zenodo_zarrs)
    parallelization_list = _init_group_by_well_for_multiplexing(
        zarr_urls=zarr_urls,
        zarr_dir="",
        reference_acquisition=0,
    )
    for img in parallelization_list["parallelization_list"]:
        zarr_url = img["zarr_url"]
        init_args = img["init_args"]
        label_name = "org_from_nuc"
        roi_table = "org_ROI_table_from_nuc"

        spherical_harmonics_from_labelimage(
            zarr_url=img["zarr_url"],
            init_args=init_args,
            label_name=label_name,
            roi_table=roi_table,
            lmax=2,
            save_mesh=True,
            save_reconstructed_mesh=True,
        )

        # check that 3 mesh files were written
        output_mesh_path = f"{zarr_url}/meshes/{roi_table}_shaics"
        output_mesh_path_reconstructed = (
            f"{zarr_url}/meshes/{roi_table}_shaics_reconstructed"
        )
        assert len(os.listdir(output_mesh_path)) == 3
        assert len(os.listdir(output_mesh_path_reconstructed)) == 3

        # check that first calculated spherical harmonic is correct
        output_table_path = f"{zarr_url}/tables/{label_name}_harmonics"
        output = ad.read_zarr(output_table_path).to_df().to_numpy()
        assert_almost_equal(
            output[:, 0], test_sphr_harmonics_from_labelimg_expected_output, decimal=5
        )


def test_scmultiplex_mesh_measurements(linking_zenodo_zarrs, name=name_3d):
    zarr_urls = select_zarr_urls(name, linking_zenodo_zarrs)
    parallelization_list = _init_group_by_well_for_multiplexing(
        zarr_urls=zarr_urls,
        zarr_dir="",
        reference_acquisition=0,
    )
    for img in parallelization_list["parallelization_list"]:
        zarr_url = img["zarr_url"]
        init_args = img["init_args"]
        mesh_name = "org_from_nuc"
        roi_table = "org_ROI_table_from_nuc"
        output_table_name = "mesh_features"

        scmultiplex_mesh_measurements(
            zarr_url=img["zarr_url"],
            init_args=init_args,
            mesh_name=mesh_name,
            roi_table=roi_table,
            output_table_name=output_table_name,
            save_hulls=True,
            calculate_curvature=True,
            calculate_harmonics=True,
            lmax=2,
            translate_to_origin=True,
            save_reconstructed_mesh=True,
        )

        # check that 3 mesh files were written for convex hull
        output_mesh_path_chull = f"{zarr_url}/meshes/{mesh_name}_convex_hull"
        assert len(os.listdir(output_mesh_path_chull)) == 3

        # check that all extracted features are correct for first organoid
        output_table_path = f"{zarr_url}/tables/{output_table_name}"
        output = ad.read_zarr(output_table_path).to_df().to_numpy()
        assert_almost_equal(
            output[0, :], test_scmultiplex_mesh_measurements_expected_output, decimal=4
        )

        # check that 3 mesh files were written for reconstructed harmonics
        output_mesh_path_reconstructed = f"{zarr_url}/meshes/{mesh_name}_reconstructed"
        assert len(os.listdir(output_mesh_path_reconstructed)) == 3

        # check that first calculated spherical harmonic is correct
        output_table_path = f"{zarr_url}/tables/{output_table_name}_harmonics"
        output = ad.read_zarr(output_table_path).to_df().to_numpy()
        print("output[!!!!!!!!!!!!!!!!0, :]", output[:, 0])
        assert_almost_equal(
            output[:, 0], test_sphr_harmonics_from_mesh_expected_output, decimal=4
        )
