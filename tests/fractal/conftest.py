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
    DOI = "10.5281/zenodo.10519143"
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


# @pytest.fixture(scope="function")
# def zenodo_zarr_metadata(testdata_path: Path):
#     metadata_3D = {
#         "plate": ["plate.zarr"],
#         "well": ["plate.zarr/B/03"],
#         "image": ["plate.zarr/B/03/0/"],
#         "num_levels": 6,
#         "coarsening_xy": 2,
#         "original_paths": [str(testdata_path / "10_5281_zenodo_7059515/")],
#         "image_extension": "png",
#     }

#     metadata_2D = {
#         "plate": ["plate.zarr"],
#         "well": ["plate_mip.zarr/B/03/"],
#         "image": ["plate_mip.zarr/B/03/0/"],
#         "num_levels": 6,
#         "coarsening_xy": 2,
#         "original_paths": [str(testdata_path / "10_5281_zenodo_7059515/")],
#         "image_extension": "png",
#         "replicate_zarr": {
#             "suffix": "mip",
#             "sources": {"plate_mip": "/this/should/not/be/used/"},
#         },
#     }

#     return [metadata_3D, metadata_2D]
