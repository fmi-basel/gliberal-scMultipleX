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
import os
from pathlib import Path
from typing import Optional

import numpy as np
from ngio import open_ome_zarr_container
from ngio.utils import ngio_logger
from pydantic import validate_call

from scmultiplex.fractal.fractal_helper_functions import (
    roi_to_pixel_slices,
    save_new_label_image_with_overlap,
)
from scmultiplex.linking.NucleiLinkingFunctions import remove_labels
from scmultiplex.meshing.FilterFunctions import min_nonzero_label
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
    parent_roi_table_name: str = "org_ROI_table_linked",
    new_child_label_name: Optional[str] = None,
    filter_children_by_volume: bool = True,
    child_volume_filter_threshold: float = 0.05,
    repair_uint16_clipped_labels: bool = False,
    remove_nonsurface_labels: bool = False,
) -> None:
    """

    Clean up debris in label images. Remove labels that are smaller than specified volume threshold, save
    as new label image and ROI table.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            Refers to the zarr_url of the reference acquisition.
            (standard argument for Fractal tasks, managed by Fractal server).
        child_label_name: Label name child objects (e.g. nuclei) to be cleaned.
        parent_roi_table_name: Name of the ROI table used to iterate over objects and load object regions. This is the
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
        remove_nonsurface_labels: If true, remove child labels that are not on organoid surface, as
            determined by Annotate_mesh_by_child_features" task. The values to be removed are loaded from disk.

    """
    ome_zarr = open_ome_zarr_container(zarr_url)

    # Load ROI tables
    parent_roi_table = ome_zarr.get_table(
        parent_roi_table_name, check_type="generic_roi_table"
    )
    parent_label_name = Path(parent_roi_table._meta.region.path).name

    # Load label image to clean
    masked_image = ome_zarr.get_masked_label(
        label_name=child_label_name, masking_label_name=parent_label_name
    )

    # Pixel sizes as list: [z,y,x]
    pixel_size = masked_image.pixel_size
    spacing = np.array([pixel_size.z, pixel_size.y, pixel_size.x])

    logger.info(
        f"Running for {zarr_url=}. \n"
        f"Cleaning {child_label_name=} with masking from {parent_label_name=}"
    )

    if new_child_label_name is None:
        output_label_name = f"{child_label_name}_cleaned"
    else:
        output_label_name = new_child_label_name

    output_roi_table_name = f"{output_label_name}_ROI_table"

    last_max_value = 0
    hit_uint16_max = False
    detected_clipped_values = False
    start_label = 65536

    # Initialize zarr with NGIO
    ome_zarr.derive_label(name=output_label_name, overwrite=True)

    if remove_nonsurface_labels:
        # Load label info to remove
        registration_save_path = os.path.join(
            zarr_url,
            "registration",
            "nonsurface_labels",
            "nonsurface_labels_to_remove.npz",
        )
        try:
            npz_data = np.load(registration_save_path)
        except FileNotFoundError or OSError:
            raise ValueError(
                f"File {registration_save_path} not found. Make sure you have run "
                f"annotate_mesh_by_features first. "
            )
        # convert to dictionary where key is organoid label (str), value is list of label ids (ints) to remove
        nonsurface_dict = {k: npz_data[k].tolist() for k in npz_data}

    # For each object in input ROI table...
    for roi in parent_roi_table.rois():

        label_string = roi.name
        label_int = int(label_string)
        logger.info(f"Processing parent ROI label {label_string}")

        # Load label image of label_name object as numpy array
        seg = masked_image.get_roi_masked(label=label_int)

        # Only proceed if labelmap is not empty
        if np.amax(seg) == 0:
            logger.warning(
                f"Skipping object ID {label_string}. Label image contains no labeled objects."
            )
            # Skip this object
            continue

        ##############
        # Repair uint16 clipped labels  ###
        ##############
        # TODO: This assumes that order of loaded objects follows order of segmentation, so that child label IDs are
        #  monotonically increasing for each parent ROI.

        if repair_uint16_clipped_labels:

            max_label = np.amax(seg)
            min_label = min_nonzero_label(seg)

            if max_label >= 65535:
                hit_uint16_max = True  # stays True for all subsequence objects

            if min_label < last_max_value:
                # if true, this means labels are no longer monotonically increasing
                detected_clipped_values = True  # stays True for all subsequence objects

            if hit_uint16_max and detected_clipped_values:
                logger.warning(
                    f"Detected clipped label values in object {label_string}."
                )
                # relabel child labels for all subsequent objects
                relabeled_image = seg.astype(
                    np.uint32
                ).copy()  # first convert to uint32

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
                    f"Volume filtering removed {len(segids_toremove)} cell(s) from object {label_string} "
                    f"that have a volume below the calculated {np.round(volume_cutoff, 1)} pixel threshold"
                    f"\n Removed labels have a mean volume of {np.round(removed_size_mean, 1)} and are the "
                    f"label id(s): "
                    f"\n {segids_toremove}"
                )

        ##############
        # Remove nonsurface labels  ###
        ##############
        if remove_nonsurface_labels:
            # Relabel input image to remove nonsurface IDs
            try:
                segids_toremove = nonsurface_dict[label_string]
            except KeyError:
                logger.warning(
                    f"Key '{label_string}' not found in nonsurface_dict. Check ROI table input."
                )
                segids_toremove = []  # remove nothing
            datatype = seg.dtype
            if len(segids_toremove) > 0:
                seg = remove_labels(seg, np.array(segids_toremove), datatype)
                logger.info(
                    f"Removed {len(segids_toremove)} cell(s) from object {label_string} "
                    f"that are nonsurface labels with label id(s): "
                    f"\n {segids_toremove}"
                )

        # Save ROI to disk using dask _to_zarr, not ngio
        region = roi_to_pixel_slices(roi, spacing)
        save_new_label_image_with_overlap(seg, zarr_url, output_label_name, region)

    logger.info("End looping over parent objects, now building pyramids.")

    ##############
    # Build pyramid and save new masking ROI table of expanded labels ###
    ##############

    # Build pyramids
    expanded_label_img = ome_zarr.get_label(name=output_label_name)
    expanded_label_img.consolidate()
    logger.info(f"Built pyramid for the {output_label_name} label image")

    # Make ROI table from expanded label image
    masking_table = ome_zarr.build_masking_roi_table(output_label_name)
    ome_zarr.add_table(output_roi_table_name, masking_table, overwrite=True)
    logger.info(f"Saved new masking ROI table as {output_roi_table_name}")

    logger.info(
        f"End cleanup_3d_child_labels task for {zarr_url}/labels/{child_label_name}"
    )

    return


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=cleanup_3d_child_labels,
        logger_name=logger.name,
    )
