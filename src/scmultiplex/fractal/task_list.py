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
from fractal_tasks_core.dev.task_models import CompoundTask
from fractal_tasks_core.dev.task_models import NonParallelTask
from fractal_tasks_core.dev.task_models import ParallelTask

# executable relative to base folder to src/scmultiplex folder
# TODO: check CPU and GPU usage for each task and allocate more accurate values

TASK_LIST = [
    CompoundTask(
        name="scMultiplex Calculate Object Linking",
        executable_init="fractal/_image_based_registration_hcs_init.py",
        executable="fractal/calculate_object_linking.py",
        meta_init={"cpus_per_task": 1, "mem": 1000},
        meta={"cpus_per_task": 4, "mem": 16000},
    ),
    CompoundTask(
        name="scMultiplex Calculate Linking Consensus",
        executable_init="fractal/_init_group_by_well_for_multiplexing.py",
        executable="fractal/calculate_linking_consensus.py",
        meta_init={"cpus_per_task": 1, "mem": 1000},
        meta={"cpus_per_task": 4, "mem": 16000},
    ),
    CompoundTask(
        name="scMultiplex Relabel by Linking Consensus",
        executable_init="fractal/_image_based_registration_hcs_allrounds_init.py",
        executable="fractal/relabel_by_linking_consensus.py",
        meta_init={"cpus_per_task": 1, "mem": 1000},
        meta={"cpus_per_task": 4, "mem": 16000},
    ),
    CompoundTask(
        name="scMultiplex Calculate Platymatch Registration",
        executable_init="fractal/_image_based_registration_hcs_init.py",
        executable="fractal/calculate_platymatch_registration.py",
        meta_init={"cpus_per_task": 1, "mem": 1000},
        meta={"cpus_per_task": 4, "mem": 16000},
    ),
    CompoundTask(
        name="scMultiplex Surface Mesh",
        executable_init="fractal/_init_group_by_well_for_multiplexing.py",
        executable="fractal/surface_mesh.py",
        meta_init={"cpus_per_task": 1, "mem": 1000},
        meta={"cpus_per_task": 4, "mem": 16000},
    ),
    ParallelTask(
        name="scMultiplex Feature Measurements",
        executable="tasks/scmultiplex_feature_measurements.py",
        meta={"cpus_per_task": 1, "mem": 4000},
    ),
]