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
Applies the multiplexing translation to all ROI tables
"""
import logging

from fractal_tasks_core.tasks.init_group_by_well_for_multiplexing import (
    init_group_by_well_for_multiplexing,
)
from pydantic import validate_call

logger = logging.getLogger(__name__)


@validate_call
def _init_group_by_well_for_multiplexing(
    *,
    # Fractal parameters
    zarr_urls: list[str],
    zarr_dir: str,
    # Core parameters
    reference_acquisition: int = 0,
) -> dict[str, list[str]]:
    """
    Finds images for all acquisitions per well.

    Returns the parallelization_list to run `find_registration_consensus`.

    Args:
        zarr_urls: List of paths or urls to the individual OME-Zarr image to
            be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        zarr_dir: path of the directory where the new OME-Zarrs will be
            created. Not used by this task.
            (standard argument for Fractal tasks, managed by Fractal server).
        reference_acquisition: Which acquisition to register against. Uses the
            OME-NGFF HCS well metadata acquisition keys to find the reference
            acquisition.
    """

    # reference round is given as zarr url, single round for the zarr_url
    # all rounds are given as init_args
    return init_group_by_well_for_multiplexing(
        zarr_urls=zarr_urls,
        zarr_dir=zarr_dir,
        reference_acquisition=reference_acquisition,
    )


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=_init_group_by_well_for_multiplexing,
        logger_name=logger.name,
    )
