# Copyright 2024 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Tommaso Comparin <tommaso.comparin@exact-lab.it>
# Joel Lüthi <joel.luethi@uzh.ch>
# Nicole Repina <nicole.repina@fmi.ch>
#
# This file is part of Fractal and was originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
"""
Select multiple multiplexing rounds for processing. Can also select single round, e.g. [0]
"""
import logging
from typing import Any

from fractal_tasks_core.utils import create_well_acquisition_dict
from pydantic import validate_call

logger = logging.getLogger(__name__)


@validate_call
def init_select_many_rounds(
    *,
    # Fractal parameters
    zarr_urls: list[str],
    zarr_dir: str,
    # Core parameters
    select_acquisitions: list[int],
) -> dict[str, list[dict[str, Any]]]:
    """
    Finds images for desired acquisition per well.

    Returns the parallelization_list.

    Args:
        zarr_urls: List of paths or urls to the individual OME-Zarr image to
            be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        zarr_dir: path of the directory where the new OME-Zarrs will be
            created. Not used by this task.
            (standard argument for Fractal tasks, managed by Fractal server).
        select_acquisitions: List of rounds to which correction should be applied, list of integers.
    """
    logger.info(f"Running `init_select_many_rounds` for {zarr_urls=}")
    image_groups = create_well_acquisition_dict(zarr_urls)

    # Create the parallelization list
    parallelization_list = []
    for key, image_group in image_groups.items():

        # Create a parallelization list entry for selected image
        for acquisition, zarr_url in image_group.items():
            if acquisition in select_acquisitions:
                parallelization_list.append(
                    dict(
                        zarr_url=zarr_url,
                        init_args=dict(),
                    )
                )

    return dict(parallelization_list=parallelization_list)


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=init_select_many_rounds,
        logger_name=logger.name,
    )
