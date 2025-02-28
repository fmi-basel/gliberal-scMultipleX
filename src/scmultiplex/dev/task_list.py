# Copyright 2024 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Joel Lüthi <joel.luethi@uzh.ch>
#
# This file is part of Fractal and was originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
"""
Fractal task list.
"""
from fractal_tasks_core.dev.task_models import CompoundTask, ParallelTask

# executable relative to base folder to src/scmultiplex folder
# TODO: check CPU and GPU usage for each task and allocate more accurate values

TASK_LIST = [
    CompoundTask(
        name="scMultiplex Calculate Object Linking",
        executable_init="fractal/_image_based_registration_hcs_init.py",
        executable="fractal/calculate_object_linking.py",
        meta_init={"cpus_per_task": 1, "mem": 1000},
        meta={"cpus_per_task": 4, "mem": 16000},
        category="Registration",
        modality="HCS",
        tags=["multiplexing", "2D"],
        docs_info="file:task_info/calculate_object_linking.md",
    ),
    CompoundTask(
        name="scMultiplex Calculate Linking Consensus",
        executable_init="fractal/_init_group_by_well_for_multiplexing.py",
        executable="fractal/calculate_linking_consensus.py",
        meta_init={"cpus_per_task": 1, "mem": 1000},
        meta={"cpus_per_task": 4, "mem": 16000},
        category="Registration",
        modality="HCS",
        tags=["multiplexing", "2D", "3D"],
        docs_info="file:task_info/calculate_linking_consensus.md",
    ),
    CompoundTask(
        name="scMultiplex Relabel by Linking Consensus",
        executable_init="fractal/_image_based_registration_hcs_allrounds_init.py",
        executable="fractal/relabel_by_linking_consensus.py",
        meta_init={"cpus_per_task": 1, "mem": 1000},
        meta={"cpus_per_task": 4, "mem": 64000},
        category="Registration",
        modality="HCS",
        tags=["multiplexing", "2D", "3D"],
        docs_info="file:task_info/relabel_by_linking_consensus.md",
    ),
    CompoundTask(
        name="scMultiplex Calculate Platymatch Registration",
        executable_init="fractal/_image_based_registration_hcs_init.py",
        executable="fractal/calculate_platymatch_registration.py",
        meta_init={"cpus_per_task": 1, "mem": 1000},
        meta={"cpus_per_task": 4, "mem": 16000},
        category="Registration",
        modality="HCS",
        tags=["multiplexing", "3D"],
        docs_info="file:task_info/calculate_platymatch_registration.md",
    ),
    CompoundTask(
        name="scMultiplex Surface Mesh Multiscale",
        executable_init="fractal/_init_group_by_well_for_multiplexing.py",
        executable="fractal/surface_mesh_multiscale.py",
        meta_init={"cpus_per_task": 1, "mem": 1000},
        meta={"cpus_per_task": 4, "mem": 16000},
        category="Image Processing",
        modality="HCS",
        tags=["3D", "mesh"],
        docs_info="file:task_info/surface_mesh_multiscale.md",
    ),
    CompoundTask(
        name="scMultiplex Segment by Intensity Threshold",
        executable_init="fractal/init_select_multiplexing_round.py",
        executable="fractal/segment_by_intensity_threshold.py",
        meta_init={"cpus_per_task": 1, "mem": 1000},
        meta={"cpus_per_task": 4, "mem": 16000},
        category="Segmentation",
        modality="HCS",
        tags=["Classical segmentation", "3D"],
        docs_info="file:task_info/segment_by_intensity_threshold.md",
    ),
    CompoundTask(
        name="scMultiplex Spherical Harmonics from Label Image",
        executable_init="fractal/_init_group_by_well_for_multiplexing.py",
        executable="fractal/spherical_harmonics_from_labelimage.py",
        meta_init={"cpus_per_task": 1, "mem": 1000},
        meta={"cpus_per_task": 4, "mem": 16000},
        category="Measurement",
        modality="HCS",
        tags=["3D"],
        docs_info="file:task_info/spherical_harmonics_from_label_image.md",
    ),
    CompoundTask(
        name="scMultiplex Mesh Measurements",
        executable_init="fractal/_init_group_by_well_for_multiplexing.py",
        executable="fractal/scmultiplex_mesh_measurements.py",
        meta_init={"cpus_per_task": 1, "mem": 1000},
        meta={"cpus_per_task": 4, "mem": 16000},
        category="Measurement",
        modality="HCS",
        tags=["3D", "mesh", "morphology"],
        docs_info="file:task_info/mesh_measurements.md",
    ),
    ParallelTask(
        name="scMultiplex Feature Measurements",
        executable="fractal/scmultiplex_feature_measurements.py",
        meta={"cpus_per_task": 4, "mem": 16000},
        category="Measurement",
        tags=["regionprops", "morphology", "intensity"],
        docs_info="file:task_info/feature_measurements.md",
    ),
    CompoundTask(
        name="scMultiplex Expand Labels",
        executable_init="fractal/init_select_many_rounds.py",
        executable="fractal/expand_labels.py",
        meta={"cpus_per_task": 4, "mem": 16000},
        category="Image Processing",
        tags=["2D", "3D"],
        docs_info="file:task_info/expand_labels.md",
    ),
    CompoundTask(
        name="scMultiplex Calculate Z-Illumination Correction",
        executable_init="fractal/init_select_multiplexing_round.py",
        executable="fractal/calculate_z_illumination_correction.py",
        meta_init={"cpus_per_task": 1, "mem": 1000},
        meta={"cpus_per_task": 4, "mem": 16000},
        category="Image Processing",
        modality="HCS",
        tags=["3D", "illumination correction"],
        docs_info="file:task_info/apply_z_illumination_correction.md",
        input_types=dict(z_illum_corrected=False, is_3D=True),
    ),
    CompoundTask(
        name="scMultiplex Apply Z-Illumination Correction",
        executable_init="fractal/init_select_illumination_round.py",
        executable="fractal/apply_z_illumination_correction.py",
        meta_init={"cpus_per_task": 1, "mem": 1000},
        meta={"cpus_per_task": 4, "mem": 16000},
        category="Image Processing",
        modality="HCS",
        tags=["3D", "illumination correction"],
        docs_info="file:task_info/apply_z_illumination_correction.md",
        input_types=dict(z_illum_corrected=False, is_3D=True),
        output_types=dict(z_illum_corrected=True, is_3D=True),
    ),
]
