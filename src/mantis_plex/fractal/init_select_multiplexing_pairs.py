# Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Tommaso Comparin <tommaso.comparin@exact-lab.it>
# Joel Lüthi <joel.luethi@uzh.ch>
#
# This file is part of Fractal and was originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
"""
Initializes the parallelization list for registration in HCS plates.
"""
import logging
from typing import Any

from fractal_tasks_core.tasks.image_based_registration_hcs_init import (
    image_based_registration_hcs_init,
)
from pydantic import validate_call

logger = logging.getLogger(__name__)


@validate_call
def init_select_multiplexing_pairs(
    *,
    # Fractal parameters
    zarr_urls: list[str],
    zarr_dir: str,
    # Core parameters
    reference_acquisition: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    """
    Initialized calculate registration task

    This task prepares a parallelization list of all zarr_urls that need to be
    used to calculate the registration between acquisitions (all zarr_urls
    except the reference acquisition vs. the reference acquisition).
    This task only works for HCS OME-Zarrs for 2 reasons: Only HCS OME-Zarrs
    currently have defined acquisition metadata to determine reference
    acquisitions. And we have only implemented the grouping of images for
    HCS OME-Zarrs by well (with the assumption that every well just has 1
    image per acqusition).

    Args:
        zarr_urls: List of paths or urls to the individual OME-Zarr image to
            be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        zarr_dir: path of the directory where the new OME-Zarrs will be
            created. Not used by this task.
            (standard argument for Fractal tasks, managed by Fractal server).
        reference_acquisition: Which acquisition to register against. Needs to
            match the acquisition metadata in the OME-Zarr image.

    Returns:
        task_output: Dictionary for Fractal server that contains a
            parallelization list.
    """

    # zarr url is the cycle x image, will have a list of non-reference rounds
    # in init args you have the reference zarr url
    return image_based_registration_hcs_init(
        zarr_urls=zarr_urls,
        zarr_dir=zarr_dir,
        reference_acquisition=reference_acquisition,
    )


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=init_select_multiplexing_pairs,
        logger_name=logger.name,
    )
