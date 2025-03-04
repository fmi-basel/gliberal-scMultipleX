# Copyright (C) 2025 Friedrich Miescher Institute for Biomedical Research

##############################################################################
#                                                                            #
# Author: Nicole Repina              <nicole.repina@fmi.ch>                  #
#                                                                            #
##############################################################################
import logging
from typing import Any, Union

import numpy as np
import pandas as pd
from fractal_tasks_core.pyramids import build_pyramid
from fractal_tasks_core.roi import array_to_bounding_box_table
from pydantic import validate_call

from scmultiplex.fractal.fractal_helper_functions import (
    initialize_new_label,
    load_image_array,
    save_masking_roi_table_from_df_list,
)
from scmultiplex.meshing.LabelFusionFunctions import simple_fuse_labels

logger = logging.getLogger(__name__)


@validate_call
def fuse_touching_labels(
    *,
    # Fractal arguments
    zarr_url: str,
    # Task-specific arguments
    label_name_to_fuse: str = "org",
    connectivity: Union[int, None] = None,
) -> dict[str, Any]:
    """
    Fuse touching labels in segmentation images, in 2D or 3D. Connected components are identified during labeling
    based on the connectivity argument. For a more detailed explanation of 1- or 2- connectivity, see documentation
    of skimage.measure.label() function. When set to None (default), full connectivity (ndim of input array) is used.

    Input is segmentation image with 0 value for background. Anything above 0 is assumed to be a labeled object.
    Touching labels are labeled in numerically increasing order starting from 1 to n, where n is the number of
    connected components (objects) identified.

    This task has been tested for fusion of 2D MIP segmentation. Since fusion must occur on the full well numpy
    array loaded into memory, performance may be poor for large 3D arrays.

    Output: the fused label image is saved as a new label in zarr, with name {label_name_to_fuse}_fused. The
    new ROI table for the fused label image is saved as a masking ROI table, with name
    {label_name_to_fuse}_fused_ROI_table.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
        label_name_to_fuse: Label name of segmentation to be fused.
        connectivity: Maximum number of orthogonal hops to consider a pixel/voxel as a neighbor. Accepted values
        are ranging from 1 to input.ndim. If None, a full connectivity of input.ndim is used.
    """

    logger.info(f"Running for {zarr_url=}. \n" f"and label image {label_name_to_fuse}.")

    # always use highest resolution label
    level = 0
    label_url = f"{zarr_url}/labels/{label_name_to_fuse}"

    label_dask, ngffmeta, xycoars, pixmeta = load_image_array(label_url, level)

    output_label_name = f"{label_name_to_fuse}_fused"
    output_roi_table_name = f"{label_name_to_fuse}_fused_ROI_table"

    shape = label_dask.shape
    chunks = label_dask.chunksize

    initialize_new_label(
        zarr_url,
        shape,
        chunks,
        np.uint32,
        label_name_to_fuse,
        output_label_name,
        logger,
    )

    logger.info("Started computation to fuse labels.")

    fused_numpy, fused_dask, label_count, connectivity_comp = simple_fuse_labels(
        label_dask, connectivity
    )

    fused_dask.to_zarr(
        f"{zarr_url}/labels/{output_label_name}/0",
        overwrite=True,
        dimension_separator="/",
        return_stored=False,
        compute=True,
    )

    logger.info(
        f"Finished computation to fuse labels, found {label_count} connected components using "
        f"connectivity {connectivity_comp}."
    )

    ##############
    # Build pyramid and save new masking ROI table of expanded labels ###
    ##############
    # Starting from on-disk highest-resolution data, build and write to disk a pyramid of coarser levels
    build_pyramid(
        zarrurl=f"{zarr_url}/labels/{output_label_name}",
        overwrite=True,
        num_levels=ngffmeta.num_levels,
        coarsening_xy=ngffmeta.coarsening_xy,
        chunksize=chunks,
        aggregation_function=np.max,
    )

    logger.info(
        f"Built a pyramid for the {zarr_url}/labels/{output_label_name} label image."
    )

    bbox_df = array_to_bounding_box_table(
        fused_numpy,
        pixmeta,
    )

    bbox_table = save_masking_roi_table_from_df_list(
        [bbox_df],
        zarr_url,
        output_roi_table_name,
        output_label_name,
        overwrite=True,
    )

    logger.debug(
        pd.DataFrame(
            bbox_table.X,
            index=bbox_table.obs_vector("label"),
            columns=bbox_table.var_names,
        )
    )

    logger.info(f"End fuse_touching_labels task for {zarr_url}")

    return {}


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=fuse_touching_labels,
        logger_name=logger.name,
    )
