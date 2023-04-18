"""
Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
University of Zurich

Original authors:
Marco Franzon <marco.franzon@exact-lab.it>
Tommaso Comparin <tommaso.comparin@exact-lab.it>

This file is part of Fractal and was originally developed by eXact lab S.r.l.
<exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
Institute for Biomedical Research and Pelkmans Lab from the University of
Zurich.
"""
import os
from pathlib import Path
import json

from devtools import debug

# from fractal_tasks_core.create_ome_zarr import create_ome_zarr
# from fractal_tasks_core.yokogawa_to_ome_zarr import yokogawa_to_ome_zarr
from scmultiplex_feature_measurements import scmultiplex_measurements

# zarr_path = "/Users/joel/shares/workShareJoel/v1_fractal/fractal-demos/examples/02_cardio_small/tmp_cardio-2x2-testing/output/"
# metadata_path = "/Users/joel/shares/homeShareFractal/joel/fractal_v1/fractal-demos/examples/server/{artifacts-110}/workflow_000007_job_000006/metadata.json"
# metadata_path = "/Users/joel/shares/homeShareFractal/joel/fractal_v1/fractal-demos/examples/server/{artifacts-110}/workflow_000007_job_000006/metadata_3D.json"

zarr_path = "/Users/joel/Dropbox/Joel/FMI/Code/fractal/fractal-demos/examples/01_cardio_tiny_dataset/tmp_cardiac-tiny-scMultiplex/output/"
metadata_path = "/Users/joel/Dropbox/Joel/FMI/Code/fractal/fractal-demos/examples/server/artifacts/workflow_000015_job_000015/metadata.json" 

with open(metadata_path) as json_file:
    metadata = json.load(json_file)


input_channels = {
    # "C01": {"wavelength_id": "A01_C01"}, 
    # "C02": {"wavelength_id": "A01_C02"}, 
    # "C03": {"wavelength_id": "A02_C03"}, 
}
label_image = 'nuclei'
output_table_name = 'table_scmultiplex_2D_no_int_img'
measure_morphology = True

# scmultiplex task running on existing Zarr file:
for component in metadata["image"]:
    scmultiplex_measurements(
        input_paths=[zarr_path],
        output_path=zarr_path,
        metadata=metadata,
        component=component,
        input_ROI_table = "well_ROI_table", #"well_ROI_table", #"FOV_ROI_table",
        input_channels = input_channels,
        label_image = label_image,
        output_table_name = output_table_name,
        measure_morphology = measure_morphology,
    )

