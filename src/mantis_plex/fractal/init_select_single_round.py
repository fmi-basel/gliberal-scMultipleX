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
Select single desired multiplexing round for processing
"""
import logging

from fractal_tasks_core.utils import create_well_acquisition_dict
from pydantic import validate_call

logger = logging.getLogger(__name__)


@validate_call
def init_select_single_round(
    *,
    # Fractal parameters
    zarr_urls: list[str],
    zarr_dir: str,
    # Core parameters
    selected_acquisition: int = 0,
) -> dict[str, list[str]]:
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
        selected_acquisition: Which multiplexing round to select for processing. Uses the
            OME-NGFF HCS well metadata acquisition keys to find the round.
    """
    logger.info(f"Running `init_select_single_round` for {zarr_urls=}")
    image_groups = create_well_acquisition_dict(zarr_urls)

    # Create the parallelization list
    parallelization_list = []
    for key, image_group in image_groups.items():
        # Assert that all image groups have the reference acquisition present
        if selected_acquisition not in image_group.keys():
            raise ValueError(
                f"Processing of {selected_acquisition=} only possible if "
                "all wells have the selected acquisition present. It was not "
                f"found for well {key}."
            )

        # Create a parallelization list entry for selected image
        zarr_url_list = []
        for acquisition, zarr_url in image_group.items():
            if acquisition == selected_acquisition:
                selected_zarr_url = zarr_url

            zarr_url_list.append(zarr_url)

        parallelization_list.append(
            dict(
                zarr_url=selected_zarr_url,
                init_args=dict(zarr_url_list=zarr_url_list),
            )
        )

    return dict(parallelization_list=parallelization_list)


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=init_select_single_round,
        logger_name=logger.name,
    )
