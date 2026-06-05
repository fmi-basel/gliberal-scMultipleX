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
Initializes the parallelization list for applying tasks to all rounds, including reference round,
with reference round specified in init args.
"""
import logging
from typing import Any

from fractal_tasks_core.utils import create_well_acquisition_dict
from pydantic import validate_call

logger = logging.getLogger(__name__)


@validate_call
def init_select_all_knowing_reference(
    *,
    # Fractal parameters
    zarr_urls: list[str],
    zarr_dir: str,
    # Core parameters
    reference_acquisition: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    """
    Initialized calculate registration task

    Same as init_select_multiplexing_pairs.py only does not exclude reference round in zarr_url list;
    all rounds are processed.

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
    logger.info(f"Running init_select_all_knowing_reference for {zarr_urls=}")
    image_groups = create_well_acquisition_dict(zarr_urls)

    # Create the parallelization list
    parallelization_list = []
    for key, image_group in image_groups.items():
        # Assert that all image groups have the reference acquisition present
        if reference_acquisition not in image_group.keys():
            raise ValueError(
                f"Registration with {reference_acquisition=} can only work if "
                "all wells have the reference acquisition present. It was not "
                f"found for well {key}."
            )
        # Add all zarr_urls except the reference acquisition to the
        # parallelization list
        for acquisition, zarr_url in image_group.items():
            reference_zarr_url = image_group[reference_acquisition]
            parallelization_list.append(
                dict(
                    zarr_url=zarr_url,
                    init_args=dict(reference_zarr_url=reference_zarr_url),
                )
            )

    return dict(parallelization_list=parallelization_list)


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=init_select_all_knowing_reference,
        logger_name=logger.name,
    )
