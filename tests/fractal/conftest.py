import os
import shutil
from pathlib import Path

import pooch
import pytest

from ..conftest import *  # noqa


@pytest.fixture(scope="session")
def tiny_zenodo_zarrs(testdata_path: Path) -> list[str]:
    """
    Prepare Zarr test data from Zenodo.

    This is based on the fractal-tasks-core task conftest
    This takes care of multiple steps:

    1. Download/unzip two Zarr containers (3D and MIP) from Zenodo, via pooch
    2. Copy the two Zarr containers into tests/data
    """

    # 1 Download Zarrs from Zenodo
    DOI = "10.5281/zenodo.20559997"
    DOI_slug = DOI.replace("/", "_").replace(".", "_")
    platenames = [
        "20200812-CardiomyocyteDifferentiation14-Cycle1.zarr",
        "20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr",
    ]
    rootfolder = testdata_path / DOI_slug
    folders = [rootfolder / plate for plate in platenames]

    registry = {
        "20200812-CardiomyocyteDifferentiation14-Cycle1.zarr.zip": None,
        "20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr.zip": None,
    }
    base_url = f"doi:{DOI}"
    POOCH = pooch.create(
        pooch.os_cache("pooch") / DOI_slug,
        base_url,
        registry=registry,
        retry_if_failed=10,
        allow_updates=False,
    )

    for ind, file_name in enumerate(
        [
            "20200812-CardiomyocyteDifferentiation14-Cycle1.zarr",
            "20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr",
        ]
    ):
        # 1) Download/unzip a single Zarr from Zenodo
        file_paths = POOCH.fetch(
            f"{file_name}.zip", processor=pooch.Unzip(extract_dir=file_name)
        )
        zarr_full_path = file_paths[0].split(file_name)[0] + file_name
        print(zarr_full_path)
        folder = folders[ind]

        # 2) Copy the downloaded Zarr into tests/data
        if os.path.isdir(str(folder)):
            shutil.rmtree(str(folder))
        shutil.copytree(Path(zarr_full_path) / file_name, folder)
    return [str(f) for f in folders]


@pytest.fixture(scope="function")
def tiny_zenodo_zarrs_base_path(tiny_zenodo_zarrs) -> Path:
    """Return the path to the tiny Zenodo Zarr test data base folder."""
    return str(Path(tiny_zenodo_zarrs[0]).parent)


@pytest.fixture(scope="function")
def metadata_tiny_zenodo() -> dict[str, dict]:
    return {
        "metadata_2D": {
            "plate": ["20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr"],
            "well": ["20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr/B/03/"],
            "image": [
                "20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr/B/03/0/"
            ],
            "num_levels": 5,
            "coarsening_xy": 2,
            "image_extension": "png",
            "image_glob_patterns": None,
        },
        "metadata_3D": {
            "plate": ["20200812-CardiomyocyteDifferentiation14-Cycle1.zarr"],
            "well": ["20200812-CardiomyocyteDifferentiation14-Cycle1.zarr/B/03/"],
            "image": ["20200812-CardiomyocyteDifferentiation14-Cycle1.zarr/B/03/0/"],
            "num_levels": 5,
            "coarsening_xy": 2,
            "image_extension": "png",
            "image_glob_patterns": None,
        },
    }


@pytest.fixture(scope="function")
def column_names() -> dict[str, list]:
    return {
        "columns_2D_common": [
            "x_pos",
            "y_pos",
        ],
        "columns_2D_morphology": [
            "is_touching_border_xy",
            "imgdim_x",
            "imgdim_y",
            "area_bbox",
            "area_convhull",
            "equivDiam",
            "extent",
            "solidity",
            "axis_major_length",
            "axis_minor_length",
            "minmajAxisRatio",
            "aspectRatio_equivalentDiameter",
            "area",
            "perimeter",
            "concavity",
            "asymmetry",
            "eccentricity",
            "circularity",
            "disconnected_components",
        ],
        "columns_2D_intensity": [
            "{Ch}.intensity_mean",
            "{Ch}.intensity_max",
            "{Ch}.intensity_min",
            "{Ch}.percentile25",
            "{Ch}.percentile50",
            "{Ch}.percentile75",
            "{Ch}.percentile90",
            "{Ch}.percentile95",
            "{Ch}.percentile99",
            "{Ch}.stdev",
            "{Ch}.skew",
            "{Ch}.kurtosis",
            "{Ch}.x_pos_weighted",
            "{Ch}.y_pos_weighted",
            "{Ch}.x_massDisp",
            "{Ch}.y_massDisp",
        ],
        "columns_3D_common": [
            "x_pos",
            "y_pos",
            "z_pos",
            "z_pos_pix",
            "volume",
        ],
        "columns_3D_morphology": [
            "is_touching_border_xy",
            "imgdim_x",
            "imgdim_y",
            "area_bbox",
            "area_convhull",
            "equivDiam",
            "extent",
            "solidity",
            "axis_major_length",
            "axis_minor_length",
            "minmajAxisRatio",
            "aspectRatio_equivalentDiameter",
            "imgdim_z",
            "is_touching_border_z",
            "surface_area",
        ],
        "columns_3D_intensity": [
            "{Ch}.intensity_mean",
            "{Ch}.intensity_max",
            "{Ch}.intensity_min",
            "{Ch}.percentile25",
            "{Ch}.percentile50",
            "{Ch}.percentile75",
            "{Ch}.percentile90",
            "{Ch}.percentile95",
            "{Ch}.percentile99",
            "{Ch}.stdev",
            "{Ch}.skew",
            "{Ch}.kurtosis",
            "{Ch}.x_pos_weighted",
            "{Ch}.y_pos_weighted",
            "{Ch}.x_massDisp",
            "{Ch}.y_massDisp",
            "{Ch}.z_pos_weighted",
            "{Ch}.z_massDisp",
        ],
    }


@pytest.fixture(scope="session")
def linking_zenodo_zarrs(testdata_path: Path) -> list[str]:
    """
    Prepare Zarr test data from Zenodo.

    This is based on the fractal-tasks-core task conftest
    This takes care of multiple steps:

    1. Download/unzip two Zarr containers (3D and MIP) from Zenodo, via pooch
    2. Copy the two Zarr containers into tests/data
    """

    # 1 Download Zarrs from Zenodo
    DOI = "10.5281/zenodo.13982701"
    DOI_slug = DOI.replace("/", "_").replace(".", "_")
    platenames = [
        "220605_151046.zarr",
        "220605_151046_mip.zarr",
    ]
    rootfolder = testdata_path / DOI_slug
    folders = [rootfolder / plate for plate in platenames]

    registry = {
        "220605_151046.zarr.zip": None,
        "220605_151046_mip.zarr.zip": None,
    }
    base_url = f"doi:{DOI}"
    POOCH = pooch.create(
        pooch.os_cache("pooch") / DOI_slug,
        base_url,
        registry=registry,
        retry_if_failed=10,
        allow_updates=False,
    )

    for ind, file_name in enumerate(platenames):
        # 1) Download/unzip a single Zarr from Zenodo
        file_paths = POOCH.fetch(
            f"{file_name}.zip", processor=pooch.Unzip(extract_dir=file_name)
        )
        zarr_full_path = file_paths[0].split(file_name)[0] + file_name
        print(zarr_full_path)
        folder = folders[ind]

        # 2) Copy the downloaded Zarr into tests/data
        if os.path.isdir(str(folder)):
            shutil.rmtree(str(folder))
        shutil.copytree(Path(zarr_full_path) / file_name, folder)
    return [str(f) for f in folders]


@pytest.fixture(scope="function")
def linking_zenodo_zarrs_base_path(linking_zenodo_zarrs) -> str:
    """Return the path to the tiny Zenodo Zarr test data base folder."""
    return str(Path(linking_zenodo_zarrs[0]).parent)
