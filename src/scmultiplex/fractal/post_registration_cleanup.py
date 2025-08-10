# Copyright 2025 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Nicole Repina <nicole.repina@fmi.ch>
#

"""
Copy over reference images and tables, labels, and folders to registered images.
"""

import logging
import os
from pathlib import Path

import numpy as np
from fractal_tasks_core.tasks.io_models import InitArgsRegistration
from ngio import open_ome_zarr_container
from ngio.utils import ngio_logger
from pydantic import validate_call

# Local application imports
from scmultiplex.fractal.fractal_helper_functions import (
    copy_folder_from_zarrurl,
    extract_acq_info,
    remove_roi_table_suffix,
    roi_to_pixel_slices,
    save_new_multichannel_image_with_overlap,
)
from scmultiplex.meshing.LabelFusionFunctions import select_label
from scmultiplex.utils.ngio_utils import update_well_zattrs_with_new_image

# Configure logging
ngio_logger.setLevel("ERROR")
logger = logging.getLogger(__name__)


@validate_call
def post_registration_cleanup(
    *,
    # Fractal arguments
    zarr_url: str,
    init_args: InitArgsRegistration,
    # Task-specific arguments
    cleanup_1_register_reference_round: bool = False,
    roi_table_name: str = None,
    output_image_suffix: str = None,
    mask_output_by_parent: bool = False,
    copy_labels_and_tables: bool = False,
    overwrite_entire_container: bool = False,
    cleanup_2_copy_labels_and_tables_from_unregistered_reference: bool = False,
    unregistered_reference_zarr_image_name: str = None,
    roi_table_names_to_copy: list[str] = None,
    standardize_label_name: bool = False,
    overwrite_labels_and_tables: bool = False,
    cleanup_3_copy_folders_from_matching_unregistered_round: bool = False,
    folders_to_copy: list[str] = None,
    image_suffix_to_remove: str = "_registered",
    overwrite_folders: bool = False,
):
    """
    Cleanup 1: must be run separately from Cleanup 2 & 3. Submit only unregistered reference zarr on Fractal server.
        Image data (all channels) from selected ROIs are copied from this reference, e.g. "0_fused" to a new reference
        image e.g. "0_fused_registered". Inputs should mirror Apply Warpfield Registration task inputs.
    Cleanup 2: tables and labels from the e.g. "0_fused" unregistered reference round can be copied to the
        subsequent registered moving rounds e.g. "0_fused_registered", "1_fused_registered", etc.
        Submit all registered rounds to Fractal server.
    Cleanup 3: misc folders e.g. ["plots", "registration"] can be copied from the unregistered rounds
        e.g. "0_fused", "1_fused", etc, to the corresponding registered round e.g.
        "0_fused_registered", "1_fused_registered", etc. Submit all registered rounds to Fractal server.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        init_args: Intialization arguments provided by
            `init_select_all_knowing_reference`. They contain the reference_zarr_url information.
            (standard argument for Fractal tasks, managed by Fractal server).
        cleanup_1_register_reference_round: If True, the reference round is selected from the submitted zarr urls
            and is copied to a new image with the _{output_image_suffix} appended. E.g. Image data (all channels) of
            selected ROIs are copied from "0_fused" to a new "0_fused_registered" round.
        roi_table_name: Name of masking ROI table to be used for ROI loading. Typically, this is the
            same roi table used in the apply_warpfield_registration task to load region pairs between rounds. This roi
            table is used to select ROIs that are copied over to new registered reference zarr.
        output_image_suffix: This suffix is added to the new OME-Zarr image that contains
            the registered corrected images, e.g. 'registered'. Fractal image list updated accordingly. Typically
            matches apply_warpfield_registration input.
        mask_output_by_parent: If True, the reference ROI is masked by parent object label map (from 'roi_table_name').
            Typically matches apply_warpfield_registration input.
        copy_labels_and_tables: If True, all labels and tables are copied to the new registered reference OME-Zarr
            image from the reference zarr. Typically matches apply_warpfield_registration input.
        overwrite_entire_container: If True, overwrite existing OME-Zarr image
            (including clearing all existing labels and tables!), i.e. the new OME-Zarr image with _output_image_suffix.
        cleanup_2_copy_labels_and_tables_from_unregistered_reference: If True, copy over specified tables and
            corresponding labels from the provided 'unregistered_reference_zarr_image_name' to all submitted rounds.
            E.g. tables and labels from the "0_fused" round can be copied to the rounds "0_fused_registered",
            "1_fused_registered", etc
        unregistered_reference_zarr_image_name: Name of source OME-Zarr image (unregistered reference round) from which
            data should be copied, e.g. '0_fused_zillum'.
        roi_table_names_to_copy: List of ROI table names present in unregistered_reference_zarr_image_name to copy
            over. The corresponding label of masking ROI tables are also copied over.
        standardize_label_name: If True, rename label to match the ROI table. The string "_ROI_table" is
            removed from table name to generate label name. E.g. if ROI table is called "org_linked_ROI_table", the
            label will be names "org_linked" for all rounds. If False, the label is saved using the name given in
            ROI table .zattrs metadata.
        overwrite_labels_and_tables: If True, overwrite labels and tables with same name in registered zarr image.
        cleanup_3_copy_folders_from_matching_unregistered_round: If True, copy specified folder from matching
            unregistered image. E.g. ["plots", "registration"] folders can be copied from the "1_fused" round to the
            "1_fused_registered" round. If only the reference round (e.g. "0_fused_registered") is submitted
            via the Fractal web interface, it is also possible to copy ["meshes"] from e.g. "0_fused" to the
            "0_fused_registered" round.
        folders_to_copy: List of folder names to copy over, e.g. ["registration"]
        image_suffix_to_remove: Image suffix to remove from submitted registered image to generate the unregistered
            zarr, e.g. to copy folders from "1_fused" to "1_fused_registered", the suffix is "_registered"
        overwrite_folders: If True, overwrite folders with same name in registered zarr image
    """
    # iterate over each round, not pairwise loading
    reference_zarr_url = init_args.reference_zarr_url
    ref_round_id = extract_acq_info(reference_zarr_url)
    current_round_id = extract_acq_info(zarr_url)

    image_list_updates = {}

    ##########################################
    # Run cleanup 1 only for reference zarr url: make copy of ref round with only linked ROIs
    ##########################################
    if cleanup_1_register_reference_round and current_round_id == ref_round_id:
        # perform user input checks
        if roi_table_name is None:
            raise ValueError(
                "`roi_table_name` must be provided if cleanup_1_register_reference_round is set to True."
            )
        if output_image_suffix is None:
            raise ValueError(
                "`output_image_suffix` must be provided if cleanup_1_register_reference_round is set to True."
            )

        # make a _registered copy of reference round that include only linked fields
        # here zarr_url points to the reference round
        image_name = os.path.basename(zarr_url)
        new_image_name = f"{image_name}_{output_image_suffix}"

        # Open the ome-zarr image container
        ome_zarr = open_ome_zarr_container(zarr_url)

        new_zarr_url = os.path.join(os.path.dirname(zarr_url), new_image_name)

        roi_table = ome_zarr.get_table(roi_table_name, check_type="generic_roi_table")

        label_name = Path(roi_table._meta.region.path).name

        # Load images from container
        image = ome_zarr.get_masked_image(masking_label_name=label_name)
        label_image = ome_zarr.get_masked_label(
            label_name=label_name, masking_label_name=label_name
        )

        # Pixel sizes as list: [z,y,x]
        pixel_size = image.pixel_size
        spacing = np.array([pixel_size.z, pixel_size.y, pixel_size.x])

        # Update well metadata (.zattrs) to add new image, only if does not already exist
        update_well_zattrs_with_new_image(
            zarr_url=zarr_url,
            new_image_name=new_image_name,
            acquisition_id=ref_round_id,
        )

        # Derive the new reference image (e.g. 0_registered) from the reference image (e.g. 0)
        new_ome_zarr = ome_zarr.derive_image(
            store=new_zarr_url,
            copy_labels=copy_labels_and_tables,
            copy_tables=copy_labels_and_tables,
            overwrite=overwrite_entire_container,
        )
        # Copy over only registered ROIs
        for roi in roi_table.rois():
            label_string = roi.name
            label_int = int(label_string)
            logger.info(f"Processing ROI label {label_string} for all channels")
            try:
                img_np = image.get_roi(label=label_int)  # load all channels: c,z,y,x
            except KeyError as e:
                logger.warning(
                    f"Reference image does not contain matching ROI. Skipping reference ROI {roi}. Error: {e}"
                )
                continue

            if mask_output_by_parent:
                # Load reference label image
                masking_label = label_image.get_roi(label=label_int)
                masking_label = select_label(
                    masking_label, label_string
                )  # binarize mask

            # Optionally mask output by parent after transformation
            result = np.empty_like(img_np)
            for c, channel_img_np in enumerate(img_np):
                if mask_output_by_parent:
                    if channel_img_np.shape != masking_label.shape:
                        raise ValueError(
                            f"Registration output image shape {channel_img_np.shape} does "
                            f"not match masking label shape {masking_label.shape}"
                        )
                    channel_img_np = channel_img_np * masking_label  # mask

                result[c] = channel_img_np  # store in the same channel position

            # save ROI to disk using dask _to_zarr, not ngio
            region = roi_to_pixel_slices(roi, spacing)
            save_new_multichannel_image_with_overlap(
                result, new_zarr_url, region, apply_to_all_channels=True
            )

            logger.info(f"Wrote region {label_string} to level-0 zarr image.")

        # Build pyramids
        new_moving_image = new_ome_zarr.get_image()
        new_moving_image.consolidate()

        # Update Fractal image list
        image_list_updates = dict(
            image_list_updates=[dict(zarr_url=new_zarr_url, origin=zarr_url)]
        )

    ##########################################
    # Run cleanup 2 for all selected rounds (ref and moving), copy over FROM REFERENCE ROUND:
    # selected ROI tables
    # selected corresponding label images if ROI table is a masking ROI (match label name to ROI table for consistency)
    # all selected folders
    ##########################################

    if cleanup_2_copy_labels_and_tables_from_unregistered_reference:
        # Open the ome-zarr image container from which to copy
        source_zarr_url = os.path.join(
            os.path.dirname(zarr_url), unregistered_reference_zarr_image_name
        )
        source_ome_zarr = open_ome_zarr_container(source_zarr_url)
        target_zarr_image_name = os.path.basename(zarr_url)
        target_ome_zarr = open_ome_zarr_container(zarr_url)

        label_names_to_copy = []
        new_label_names = []

        # Copy tables
        if roi_table_names_to_copy is not None:
            for table_name in roi_table_names_to_copy:
                table = source_ome_zarr.get_table(table_name)

                logger.info(
                    f"Copying table {table_name} from image {unregistered_reference_zarr_image_name} to "
                    f"{target_zarr_image_name}."
                )

                target_ome_zarr.add_table(
                    table_name, table, overwrite=overwrite_labels_and_tables
                )
                # Make list of labels to copy
                if table.type() == "masking_roi_table":
                    # get name of corresponding label as stored in table metadata
                    label_names_to_copy.append(Path(table._meta.region.path).name)
                    if standardize_label_name:
                        new_label_names.append(remove_roi_table_suffix(table_name))

        # Copy labels
        for i, label_name in enumerate(label_names_to_copy):
            if standardize_label_name:
                new_label_name = new_label_names[i]
            else:
                new_label_name = label_name

            logger.info(
                f"Copying label {label_name} from image {unregistered_reference_zarr_image_name} to "
                f"{target_zarr_image_name} as {new_label_name}."
            )

            label_image = source_ome_zarr.get_label(name=label_name)
            new_label = target_ome_zarr.derive_label(
                new_label_name, overwrite=overwrite_labels_and_tables
            )
            label_image_dask = label_image.get_array(mode="dask")
            new_label.set_array(label_image_dask)
            new_label.consolidate()

    if cleanup_3_copy_folders_from_matching_unregistered_round:

        target_zarr_image_name = os.path.basename(zarr_url)
        unregistered_source_image_name = target_zarr_image_name.rstrip(
            image_suffix_to_remove
        )
        source_zarr_url = os.path.join(
            os.path.dirname(zarr_url), unregistered_source_image_name
        )

        # Copy misc folders from source zarr, e.g. meshes, etc
        if folders_to_copy is not None:
            for folder in folders_to_copy:
                logger.info(
                    f"Copying folder {folder} from image {unregistered_source_image_name} to "
                    f"{target_zarr_image_name}."
                )

                copy_folder_from_zarrurl(
                    source_zarr_url,
                    zarr_url,
                    folder_name=folder,
                    overwrite=overwrite_folders,
                )

    logger.info(f"End post_registration_cleanup task for {zarr_url=}")

    return image_list_updates


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=post_registration_cleanup,
        logger_name=logger.name,
    )
