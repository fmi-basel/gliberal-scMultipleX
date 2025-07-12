# Copyright (C) 2025 Friedrich Miescher Institute for Biomedical Research

##############################################################################
#                                                                            #
# Author: Nicole Repina              <nicole.repina@fmi.ch>                  #
#                                                                            #
##############################################################################


"""
Remove debris based on volume filtering from 3D segmentation.
"""
import logging
from typing import Any, Optional

import anndata as ad
import dask.array as da
import ngio
import numpy as np
from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.pyramids import build_pyramid
from fractal_tasks_core.roi import (
    check_valid_ROI_indices,
    convert_indices_to_regions,
    convert_ROI_table_to_indices,
    load_region,
)
from ngio.utils import NgioFileNotFoundError, ngio_logger
from pydantic import validate_call

from scmultiplex.fractal.fractal_helper_functions import (
    get_zattrs,
    initialize_new_label,
)
from scmultiplex.meshing.FilterFunctions import mask_by_parent_object, min_nonzero_label
from scmultiplex.meshing.LabelFusionFunctions import filter_by_volume

ngio_logger.setLevel("ERROR")

logger = logging.getLogger(__name__)


@validate_call
def cleanup_3d_child_labels(
    *,
    # Fractal arguments
    zarr_url: str,
    # Task-specific arguments
    child_label_name: str = "nuc",
    parent_label_name: str,
    parent_roi_table: str = "org_ROI_table_linked",
    new_child_label_name: Optional[str] = None,
    filter_children_by_volume: bool = True,
    child_volume_filter_threshold: float = 0.05,
    repair_uint16_clipped_labels: bool = False,
) -> dict[str, Any]:
    """

    Clean up debris in label images. Remove labels that are smaller than specified volume threshold, save
    as new label image and ROI table.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            Refers to the zarr_url of the reference acquisition.
            (standard argument for Fractal tasks, managed by Fractal server).
        child_label_name: Label name child objects (e.g. nuclei) to be cleaned.
        parent_label_name: Label name of segmented objects that are parents of child_label_name.
            child_label_name objects will be masked and grouped by this object. For example, when parent_label_name = 'org', the
            nuclear segmentation is masked by the organoid parent and all nuclei belonging to the parent are loaded
            as a label image.
        parent_roi_table: Name of the ROI table used to iterate over objects and load object regions. This is the
            ROI table that corresponds to the parent_label_name objects.
        new_child_label_name: New label name for cleaned child objects. If left None, default
            is {child_label_name}_cleaned.
        filter_children_by_volume: if True, performing volume filtering of children to remove objects smaller
            than specified volume_filter_threshold.
        child_volume_filter_threshold: Multiplier that specifies cutoff for volumes below which nuclei are filtered out,
            float in range [0,1], e.g. 0.05 means that 5% of median of nuclear volume distribution in a given
            object is used as cutoff. Default 0.05.
        repair_uint16_clipped_labels: If child labels were clipped to uint16 during segmentation and
            there were more than 2^16 labels, the label id's above 65535 get clipped. If True, these clipped
            values get remapped to monotonically increasing values 65536, 65537, etc.

    """
    logger.info(
        f"Running for {zarr_url=}. \n"
        f"Cleaning {child_label_name=} with masking from {parent_label_name=}"
    )

    # always use highest resolution label
    level = 0

    # always overwrite
    overwrite = True

    # Lazily load zarr array for reference cycle
    # load well image as dask array e.g. for nuclear segmentation
    label_dask = da.from_zarr(f"{zarr_url}/labels/{child_label_name}/{level}")

    # Read ROIs of objects
    roi_adata = ad.read_zarr(f"{zarr_url}/tables/{parent_roi_table}")
    roi_attrs = get_zattrs(f"{zarr_url}/tables/{parent_roi_table}")

    # Read Zarr metadata
    label_ngffmeta = load_NgffImageMeta(f"{zarr_url}/labels/{child_label_name}")
    label_xycoars = (
        label_ngffmeta.coarsening_xy
    )  # need to know when building new pyramids
    label_pixmeta = label_ngffmeta.get_pixel_sizes_zyx(level=level)
    num_levels = label_ngffmeta.num_levels

    # Create list of indices for 3D ROIs spanning the entire Z direction
    # Note that this ROI list is generated based on the input ROI table; if the input ROI table is for the group_by
    # objects, then label regions will be loaded based on the group_by ROIs
    roi_idlist = convert_ROI_table_to_indices(
        roi_adata,
        level=level,
        coarsening_xy=label_xycoars,
        full_res_pxl_sizes_zyx=label_pixmeta,
    )

    check_valid_ROI_indices(roi_idlist, parent_roi_table)

    if len(roi_idlist) == 0:
        logger.warning("Well contains no objects")

    if new_child_label_name is None:
        output_label_name = f"{child_label_name}_cleaned"
    else:
        output_label_name = new_child_label_name

    output_roi_table_name = f"{output_label_name}_ROI_table"

    shape = label_dask.shape
    chunks = label_dask.chunksize

    new_label3d_array = initialize_new_label(
        zarr_url, shape, chunks, np.uint32, child_label_name, output_label_name, logger
    )

    logger.info(f"Output label path: {zarr_url}/labels/{output_label_name}/0")

    logger.info(f"Mask will have shape {shape} and chunks {chunks}")

    ##############
    # Filter nuclei by parent mask ###
    ##############

    # Load group_by object segmentation to mask child objects by parent group_by object
    # Load well image as dask array for parent objects
    groupby_dask = da.from_zarr(f"{zarr_url}/labels/{parent_label_name}/{level}")

    # Read Zarr metadata
    groupby_ngffmeta = load_NgffImageMeta(f"{zarr_url}/labels/{parent_label_name}")
    groupby_xycoars = groupby_ngffmeta.coarsening_xy
    groupby_pixmeta = groupby_ngffmeta.get_pixel_sizes_zyx(level=level)

    groupby_idlist = convert_ROI_table_to_indices(
        roi_adata,
        level=level,
        coarsening_xy=groupby_xycoars,
        full_res_pxl_sizes_zyx=groupby_pixmeta,
    )

    check_valid_ROI_indices(groupby_idlist, parent_roi_table)

    # Get labels to iterate over
    instance_key = roi_attrs["instance_key"]  # e.g. "label"

    # NGIO FIX, TEMP
    # Check that ROI_table.obs has the right column and extract label_value
    if instance_key not in roi_adata.obs.columns:
        if roi_adata.obs.index.name == instance_key:
            # Workaround for new ngio table
            roi_adata.obs[instance_key] = roi_adata.obs.index
        else:
            raise ValueError(
                f"In input ROI table, {instance_key=} "
                f" missing in {roi_adata.obs.columns=}"
            )

    roi_labels = roi_adata.obs_vector(instance_key)
    total_label_count = len(roi_labels)
    compute = True

    logger.info(
        f"Starting iteration over {total_label_count} detected objects in ROI table."
    )

    last_max_value = 0
    hit_uint16_max = False
    detected_clipped_values = False
    start_label = 65536

    # For each object in input ROI table...
    for i, obsname in enumerate(roi_adata.obs_names):

        label_str = roi_labels[i]
        logger.info(f"Processing parent object {label_str}.")
        region = convert_indices_to_regions(roi_idlist[i])

        # Load label image of label_name object as numpy array
        seg = load_region(
            data_zyx=label_dask,
            region=region,
            compute=compute,
        )

        # Mask objects by parent group_by object
        seg, parent_mask = mask_by_parent_object(
            seg, groupby_dask, groupby_idlist, i, label_str
        )
        # Only proceed if labelmap is not empty
        if np.amax(seg) == 0:
            logger.warning(
                f"Skipping object ID {label_str}. Label image contains no labeled objects."
            )
            # Skip this object
            continue

        ##############
        # Repair uint16 clipped labels  ###
        ##############
        if repair_uint16_clipped_labels:

            max_label = np.amax(seg)
            min_label = min_nonzero_label(seg)

            if max_label >= 65535:
                hit_uint16_max = True  # stays True for all subsequence objects

            if min_label < last_max_value:
                # if true, this means labels are no longer monotonically increasing
                detected_clipped_values = True  # stays True for all subsequence objects

            if hit_uint16_max and detected_clipped_values:
                logger.warning(f"Detected clipped label values in object {label_str}.")
                # relabel child labels for all subsequent objects
                relabeled_image = seg.copy()

                used_labels = np.unique(seg)  # sorted in numerically increasing order
                used_labels = used_labels[used_labels != 0]  # drop 0 background

                if max_label == 65535:
                    # this is the first object after clipping, and it has a mix of both high and low labels
                    # enumerate ensures monotonically increasing order
                    # Create mapping: old_label (key) → new_label (value)
                    # e.g. 1 -> 65536, 2->65537, etc
                    # Skip labels between last_max_value and max_label, as they were already used before clipping
                    mapping = {
                        old_label: new_label
                        for new_label, old_label in enumerate(
                            [
                                lab
                                for lab in used_labels
                                if not (last_max_value < lab <= max_label)
                            ],
                            start=start_label,
                        )
                    }
                else:
                    # for all subsequent objects, relabel all values
                    # Create mapping: old_label (key) → new_label (value)
                    # e.g. 1 -> 65536, 2->65537, etc
                    mapping = {
                        old_label: new_label
                        for new_label, old_label in enumerate(
                            used_labels, start=start_label
                        )
                    }

                # Vectorized relabeling, only relabel the values that are in the mapping dictionary
                for old_label, new_label in mapping.items():
                    relabeled_image[seg == old_label] = new_label

                start_label = (
                    max(mapping.values()) + 1
                )  # starting value for next object continues from current

                # don't bother updating last_max_value, this is not irrelevant

                seg = relabeled_image  # update seg with relabeled image

            else:
                # object is still in uint16 range, simply update the last_max_value and do not relabel
                last_max_value = max_label

        ##############
        # Perform volume filtering  ###
        ##############
        if filter_children_by_volume:
            (
                seg,
                segids_toremove,
                removed_size_mean,
                size_mean,
                volume_cutoff,
            ) = filter_by_volume(seg, child_volume_filter_threshold)

            if len(segids_toremove) > 0:
                logger.info(
                    f"Volume filtering removed {len(segids_toremove)} cell(s) from object {label_str} "
                    f"that have a volume below the calculated {np.round(volume_cutoff,1)} pixel threshold"
                    f"\n Removed labels have a mean volume of {np.round(removed_size_mean,1)} and are the "
                    f"label id(s): "
                    f"\n {segids_toremove}"
                )

        # Compute and store 0-th level to disk
        da.array(seg).to_zarr(
            url=new_label3d_array,
            region=region,
            compute=True,
        )

    logger.info("End looping over parent objects, now building pyramids.")

    # Starting from on-disk highest-resolution data, build and write to disk a
    # pyramid of coarser levels
    build_pyramid(
        zarrurl=f"{zarr_url}/labels/{output_label_name}",
        overwrite=overwrite,
        num_levels=num_levels,
        coarsening_xy=label_xycoars,
        chunksize=chunks,
        aggregation_function=np.max,
    )

    # 6) Make tables from new label image
    zarr_url = zarr_url.rstrip("/")

    try:
        ome_zarr_container = ngio.open_ome_zarr_container(zarr_url)
    except NgioFileNotFoundError as e:
        raise ValueError(f"OME-Zarr {zarr_url} not found.") from e

    masking_table = ome_zarr_container.build_masking_roi_table(child_label_name)

    ome_zarr_container.add_table(output_roi_table_name, masking_table, overwrite=True)

    logger.info(f"Saved new masking ROI table as {output_roi_table_name}")

    new_saved_table = ad.read_zarr(f"{zarr_url}/tables/{output_roi_table_name}")
    labels = np.array(new_saved_table.obs.index)
    unique_labels, counts = np.unique(labels, return_counts=True)
    duplicates = unique_labels[counts > 1]

    if duplicates.size > 0:
        logger.error(f"Detected duplicated ROI labels: {duplicates}")
        raise ValueError(f"ROI table contains duplicate labels: {duplicates}")

    logger.info(
        f"End cleanup_3d_child_labels task for {zarr_url}/labels/{child_label_name}"
    )

    return


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=cleanup_3d_child_labels,
        logger_name=logger.name,
    )
