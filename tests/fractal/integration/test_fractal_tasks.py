import anndata as ad
import numpy as np
import os
from fractal_tasks_core.channels import ChannelInputModel
from numpy.testing import assert_almost_equal
from pathlib import Path
from scmultiplex.fractal._image_based_registration_hcs_allrounds_init import \
    _image_based_registration_hcs_allrounds_init
from scmultiplex.fractal._image_based_registration_hcs_init import _image_based_registration_hcs_init
from scmultiplex.fractal._init_group_by_well_for_multiplexing import _init_group_by_well_for_multiplexing
from scmultiplex.fractal.calculate_linking_consensus import calculate_linking_consensus
from scmultiplex.fractal.calculate_object_linking import calculate_object_linking
from scmultiplex.fractal.calculate_platymatch_registration import calculate_platymatch_registration
from scmultiplex.fractal.relabel_by_linking_consensus import relabel_by_linking_consensus
from scmultiplex.fractal.spherical_harmonics_from_labelimage import spherical_harmonics_from_labelimage
from scmultiplex.fractal.spherical_harmonics_from_mesh import spherical_harmonics_from_mesh
from scmultiplex.fractal.surface_mesh_multiscale import surface_mesh_multiscale

name_3d = '220605_151046.zarr'
name_mip = '220605_151046_mip.zarr'

test_calculate_object_linking_expected_output = np.array([[1., 1., 0.9305816],
                                                          [2., 2., 0.78365386],
                                                          [3., 3., 0.95049506]])

test_calculate_linking_consensus_expected_output = np.array([[1., 1., 0., 1.],
                                                             [2., 2., 1., 2.],
                                                             [3., 3., 2., 3.]])

test_relabel_by_linking_consensus_output_dict = {'0': np.array([[13.,  20.8,   0.,  22.533333,  21.666666, 0.6],
                                                                [13.,  90.13333,   0.,  20.8,  18.2, 0.6],
                                                                [38.133335, 103.13333,   0., 25.133333, 29.466667, 0.6]]),
                                                 '1': np.array([[49.4, 10.4, 0., 23.4, 22.533333, 0.6],
                                                                [51.133335, 81.46667, 0., 19.933332, 16.466667, 0.6],
                                                                [74.53333, 92.73333, 0., 24.266666, 29.466667, 0.6]])
                                                 }
test_calculate_platymatch_registration_output = np.array(
[[ 1.,         1.],
 [ 3.,         2.],
 [ 2.,         3.],
 [ 4.,         4.],
 [ 5.,         5.],
 [ 6.,         6.],
 [ 7.,         7.],
 [ 8.,         8.],
 [ 9.,         9.],
 [10.,        10.],
 [11.,        11.],
 [12.,        12.],
 [13.,        13.],
 [14.,        14.],
 [15.,        15.],
 [16.,        16.]])

test_sphr_harmonics_from_labelimg_expected_output = np.array([52.831512, 47.05219,  66.96421])

test_sphr_harmonics_from_mesh_expected_output = np.array([49.73446, 43.35559, 65.31301])


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
    parallelization_list = _image_based_registration_hcs_init(zarr_urls=zarr_urls,
                                                              zarr_dir='',
                                                              reference_acquisition=0, )

    for img in parallelization_list["parallelization_list"]:
        zarr_url = img['zarr_url']
        init_args = img['init_args']
        label_name = 'org'
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
    parallelization_list = _init_group_by_well_for_multiplexing(zarr_urls=zarr_urls,
                                                                zarr_dir='',
                                                                reference_acquisition=0, )

    for img in parallelization_list["parallelization_list"]:
        zarr_url = img['zarr_url']
        init_args = img['init_args']
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
    parallelization_list = _image_based_registration_hcs_allrounds_init(zarr_urls=zarr_urls,
                                                                        zarr_dir='',
                                                                        reference_acquisition=0, )
    for img in parallelization_list["parallelization_list"]:
        zarr_url = img['zarr_url']
        init_args = img['init_args']
        label_name = 'org'
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
        assert_almost_equal(output, test_relabel_by_linking_consensus_output_dict[Path(zarr_url).name], decimal=3)


def test_calculate_platymatch_registration(linking_zenodo_zarrs, name=name_3d):
    zarr_urls = select_zarr_urls(name, linking_zenodo_zarrs)
    parallelization_list = _image_based_registration_hcs_init(zarr_urls=zarr_urls,
                                                              zarr_dir='',
                                                              reference_acquisition=0, )

    for img in parallelization_list["parallelization_list"]:
        zarr_url = img['zarr_url']
        init_args = img['init_args']
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
            volume_filter_threshold=0.05,
        )

        output_table_path_affine = f"{zarr_url}/tables/{label_name_to_register}_match_table_affine"
        output_affine = ad.read_zarr(output_table_path_affine).to_df().to_numpy()
        output_table_path_ffd = f"{zarr_url}/tables/{label_name_to_register}_match_table_ffd"
        output_ffd = ad.read_zarr(output_table_path_ffd).to_df().to_numpy()
        # test that matches are correct; ignore confidence columns
        assert_almost_equal(output_affine[:, 0:2], test_calculate_platymatch_registration_output, decimal=3)
        assert_almost_equal(output_ffd[:, 0:2], test_calculate_platymatch_registration_output, decimal=3)


def test_surface_mesh(linking_zenodo_zarrs, name=name_3d):
    zarr_urls = select_zarr_urls(name, linking_zenodo_zarrs)
    parallelization_list = _init_group_by_well_for_multiplexing(zarr_urls=zarr_urls,
                                                                zarr_dir='',
                                                                reference_acquisition=0, )
    for img in parallelization_list["parallelization_list"]:
        zarr_url = img['zarr_url']
        init_args = img['init_args']
        label_name_obj = "org"
        label_name = "nuc"
        roi_table = "org_ROI_table"

        surface_mesh_multiscale(
            zarr_url=zarr_url,
            init_args=init_args,
            label_name=label_name,
            label_name_obj=label_name_obj,
            roi_table=roi_table,
            expandby_factor=0.6,
            sigma_factor=5,
            canny_threshold=0.3,
            calculate_mesh=True,
            calculate_mesh_features=True,
            calculate_curvature=True,
            polynomial_degree=30,
            passband=0.01,
            feature_angle=160,
            target_reduction=0.98,
            smoothing_iterations=2,
            save_labels=True,
        )

        output_mesh_path = f"{zarr_url}/meshes/{label_name_obj}_from_{label_name}"
        # check that 3 mesh files were written
        assert len(os.listdir(output_mesh_path)) == 3


def test_sphr_harmonics_from_labelimage(linking_zenodo_zarrs, name=name_3d):
    zarr_urls = select_zarr_urls(name, linking_zenodo_zarrs)
    parallelization_list = _init_group_by_well_for_multiplexing(zarr_urls=zarr_urls,
                                                                zarr_dir='',
                                                                reference_acquisition=0, )
    for img in parallelization_list["parallelization_list"]:
        zarr_url = img['zarr_url']
        init_args = img['init_args']
        label_name = "org_3d"
        roi_table = "org_ROI_table_3d"

        spherical_harmonics_from_labelimage(
            zarr_url=img['zarr_url'],
            init_args=img['init_args'],
            label_name=label_name,
            roi_table=roi_table,
            lmax=2,
            save_mesh=True,
            save_reconstructed_mesh=True,
        )

        # check that 3 mesh files were written
        output_mesh_path = f"{zarr_url}/meshes/{roi_table}_shaics"
        output_mesh_path_reconstructed = f"{zarr_url}/meshes/{roi_table}_shaics_reconstructed"
        assert len(os.listdir(output_mesh_path)) == 3
        assert len(os.listdir(output_mesh_path_reconstructed)) == 3

        # check that first calculated spherical harmonic is correct
        output_table_path = f"{zarr_url}/tables/{label_name}_harmonics"
        output = ad.read_zarr(output_table_path).to_df().to_numpy()
        assert_almost_equal(output[:, 0], test_sphr_harmonics_from_labelimg_expected_output, decimal=5)


def test_sphr_harmonics_from_mesh(linking_zenodo_zarrs, name=name_3d):
    zarr_urls = select_zarr_urls(name, linking_zenodo_zarrs)
    parallelization_list = _init_group_by_well_for_multiplexing(zarr_urls=zarr_urls,
                                                                zarr_dir='',
                                                                reference_acquisition=0, )
    for img in parallelization_list["parallelization_list"]:
        zarr_url = img['zarr_url']
        init_args = img['init_args']
        mesh_name = "org_from_nuc"
        roi_table = "org_ROI_table_3d"

        spherical_harmonics_from_mesh(
            zarr_url=img['zarr_url'],
            init_args=img['init_args'],
            mesh_name=mesh_name,
            roi_table=roi_table,
            lmax=2,
            translate_to_origin=True,
            save_reconstructed_mesh=True,
        )

        # check that 3 mesh files were written
        output_mesh_path_reconstructed = f"{zarr_url}/meshes/{mesh_name}_reconstructed"
        assert len(os.listdir(output_mesh_path_reconstructed)) == 3

        # check that first calculated spherical harmonic is correct
        output_table_path = f"{zarr_url}/tables/{mesh_name}_harmonics"
        output = ad.read_zarr(output_table_path).to_df().to_numpy()
        assert_almost_equal(output[:, 0], test_sphr_harmonics_from_mesh_expected_output, decimal=5)
