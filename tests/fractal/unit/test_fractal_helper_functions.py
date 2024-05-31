from scmultiplex.fractal.fractal_helper_functions import extract_acq_info


def test_extract_acq_info(linking_zenodo_zarrs):
    plate_url = linking_zenodo_zarrs[1]
    ref_url = f"{plate_url}/C/02/0"
    zarr_url = f"{plate_url}/C/02/1"
    zarr_acq = extract_acq_info(zarr_url)
    assert zarr_acq == 1
    ref_acq = extract_acq_info(ref_url)
    assert ref_acq == 0
