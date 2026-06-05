# Copyright (C) 2025 Friedrich Miescher Institute for Biomedical Research

##############################################################################
#                                                                            #
# Author: Nicole Repina              <nicole.repina@fmi.ch>                  #
#                                                                            #
##############################################################################

import datetime
import logging
import os
from typing import Any

import anndata as ad
import pandas as pd
import zarr
from fractal_tasks_core.channels import ChannelInputModel
from fractal_tasks_core.tables import write_table
from fractal_tasks_core.tasks.io_models import InitArgsRegistrationConsensus
from pydantic import validate_call

from scmultiplex.fractal.fractal_helper_functions import (
    format_roi_table,
    get_zattrs,
    load_channel_image,
    load_image_array,
    load_label_rois,
    load_seg_and_raw_region,
)
from scmultiplex.illumination.calculate_correction_functions import (
    calculate_correction,
    check_zillum_correction_table,
)
from scmultiplex.meshing.LabelFusionFunctions import select_label

logger = logging.getLogger(__name__)


@validate_call
def calculate_z_illumination_correction(
    *,
    # Fractal arguments
    zarr_url: str,
    init_args: InitArgsRegistrationConsensus,
    # Task-specific arguments
    input_channels: list[ChannelInputModel],
    label_name: str = "org",
    roi_table: str = "org_ROI_table",
    percentile: int = 90,
    low_bound_for_correction: float = 0.1,
) -> dict[str, Any]:

    """
    Calculate z-illumination correction curve for each 3D object in well. Task assumes 3D object segmentation has been
    performed to identify regions of interest. Task also assumes objects were imaged with a uniform
    stain whose intensity drop-off along z-axis is representative of z-correction that needs to be applied.

    Task has been tested with input object segmentation (specified with "label_name" and "roi_table") performed in
    3D, where the z-extent of the objects has also been roughly defined. The task should also perform on 2D-based
    segmentation in 3D, but can be sensitive to debris axially above or below the object. This scenario has not been
    thoroughly tested. In the ideal case, the 3D segmentation should tightly fit the object and exclude any lumen or
    empty space; in this way, z-correction is calculated using signal from the object itself, not background or
    debris.

    Processing proceeds as follows, with masked loading of each segmented object:
    (1) The loaded raw channel image is masked by the segmentation image. For each z-slice, the intensity value at the
    user-specified percentile ("percentile" input) is computed. This generates the intensity vs. z curve, which is
    normalized to a maximum value of 1 (i.e. range 0 to 1) and used for subsequent analysis steps. This intensity
    curve measures the drop-off in intensity over z.
    (2) To identify the start and end of the object in z, it is assumed that the intensity is low in regions outside
    of the organoid. The smoothened first derivative of the intensity curve is computed and used to identify peaks
    at the start and end, which would correspond to regions of large changes in intensity (from low to high for the
    start peak, and from high to low for the end peak). These start and end values are used to crop the intensity
    curve, to use only the region of the object for model fitting.
    (3) The cropped intensity curve (intensity vs. z) is used to fit 3 models: "Polynomial", "ExponentialDecayConcave",
    "ExponentialDecayConvex". The best fit is chosen by the highest chi-sqr value. The model is used to evaluate
    the intensity at each z, which effectively smoothens the decay curve. These values are again normalized to a max
    value of 1. For z-slices below the identified object region, the first correction value is repeated until the first
    z plane. For z-slices above the identified object region, the last correction value is repeated until the end of
    the z-stack. For quality control, plots of the intensity curve and fits are saved for each object in a
    'plots' folder within the ome-zarr structure.
    (4) The identified correction values (0 to 1, where 1 is no correction, and lower values indicate stronger
    decay and thus stronger correction) are saved as an anndata table within the image. Rows of the adata table
    correspond to the object label that the curve was calculated from. Columns correspond to z-slice, where column 0 is
    the first (typically lowest) z-slice of the object. This table is to be used as input for the
    Apply Z-illumination Correction Task, where channel images are divided by the correction values.

    Output is anndata table of correction values, with dimensions object x z-slice (row x col). It contains an obs
     column 'label' which stores the object label as string. The table is saved per channel, with name
     "zillum_{channel}", where channel is the wavelength id or label of the channel used to calculate the
     correction values. For quality control, plots of the intensity curve and fits are saved for each object in a
    'plots' folder within the ome-zarr structure. Within the plots folder, each channel has a separate folder specified
    by {channel}_{timestamp}. In this way, if the task is rerun, the tables are overwritten but the plots of older runs
    are assigned a new timestamp so that outputs using different parameters can be compared.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed. This should be the image or multiplexing
            round that contains the uniform staining to be used for illumination correction.
        init_args: Initialization arguments provided by `init_select_multiplexing_round`.
        input_channels: list of ChannelInputModel objects, where the user specifies the
            channels for calculating correction (with wavelength id or label), within the round selected in the
            init task.
        label_name: Label name of the 3D segmentation that identifies objects in image.
        roi_table: Name of the ROI table that corresponds to label_name.
        percentile: Integer value of percentile used to calculate intensity value of each z-slice. Recommended range is
            80-90. Higher values (e.g. 99) are more sensitive to individual high-intensity in image, making the
            intensity vs. z curve less smooth. Lower values may pick up intensity of background or empty space in
            the image.
        low_bound_for_correction: Float in range 0 to 1. Correction values below this value are clipped to this
            value and are thus not allowed to go below it.
            This ensures that overcorrection does not occur, e.g. if edges of object are not detected correctly.
            Recommended to set to ~ 0.1 - 0.05 (i.e. not more than 10-20x correction factor), otherwise
            when applied to image data the pixel values can become saturated.
    """

    logger.info(
        f"Running for {zarr_url=}. \n"
        f"Calculating z-illumination correction for objects in {label_name=} with {roi_table=}."
        f"Using {percentile=} for intensity detection. "
    )

    # Always use highest resolution label
    level = 0

    # Always save table with 'zillum' prefix
    output_table_name = "zillum"

    ##############
    # Load segmentation image  ###
    ##############

    (
        label_dask,
        label_adata,
        label_idlist,
        label_ngffmeta,
        label_pixmeta,
    ) = load_label_rois(
        zarr_url,
        label_name,
        roi_table,
        level,
    )

    ##############
    # Load channel image(s)  ###
    ##############

    img_array, ngffmeta_raw, xycoars_raw, pixmeta_raw = load_image_array(
        zarr_url, level
    )

    full_z_count = img_array.shape[-3]

    # Loop over channels
    for channel in input_channels:

        # Load channel dask array and ID list
        ch_dask, ch_idlist = load_channel_image(
            channel,
            img_array,
            zarr_url,
            level,
            roi_table,
            label_adata,
            xycoars_raw,
            pixmeta_raw,
        )

        ##############
        # Iterate over objects and perform segmentation  ###
        ##############
        roi_attrs = get_zattrs(f"{zarr_url}/tables/{roi_table}")
        instance_key = roi_attrs["instance_key"]  # e.g. "label"

        # NGIO FIX, TEMP
        # Check that ROI_table.obs has the right column and extract label_value
        if instance_key not in label_adata.obs.columns:
            if label_adata.obs.index.name == instance_key:
                # Workaround for new ngio table
                label_adata.obs[instance_key] = label_adata.obs.index
            else:
                raise ValueError(
                    f"In _preprocess_input, {instance_key=} "
                    f" missing in {label_adata.obs.columns=}"
                )

        # Get labels to iterate over
        roi_labels = label_adata.obs_vector(instance_key)
        total_label_count = len(roi_labels)
        compute = True
        object_count = 0

        if channel.label is not None:
            channel_id = channel.label
        elif channel.wavelength_id is not None:
            channel_id = channel.wavelength_id
        else:
            raise ValueError("Channel could not be identified.")

        # make directory for saving z-illumination plots
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(zarr_url, "plots", f"{channel_id}_{timestamp}")

        os.makedirs(filepath, exist_ok=True)

        logger.info(
            f"Starting iteration over {total_label_count} detected objects in ROI table for channel {channel_id}."
        )

        df_correction = []

        # For each object in input ROI table...
        for i, obsname in enumerate(label_adata.obs_names):
            label_str = roi_labels[i]

            seg_numpy, raw_numpy = load_seg_and_raw_region(
                label_dask, ch_dask, label_idlist, ch_idlist, i, compute
            )

            if seg_numpy.shape != raw_numpy.shape:
                raise ValueError("Shape of label and raw images must match.")

            # Check that label exists in object
            if float(label_str) not in seg_numpy:
                raise ValueError(
                    f"Object ID {label_str} does not exist in loaded segmentation image. Does input ROI "
                    f"table match label map?"
                )

            seg_numpy = select_label(seg_numpy, label_str)

            masked_image = raw_numpy * seg_numpy

            logger.info(f"Processing object {label_str}.")

            roi_start_z = label_idlist[i][0]

            # calculate z-illumination dropoff
            row = calculate_correction(
                masked_image,
                roi_start_z,
                full_z_count,
                label_str,
                filepath,
                low_bound_for_correction,
                percentile=percentile,
            )

            if row is None:
                logger.warning(f"Skipping object {label_str}.")
            else:
                object_count += 1

                # convert to dictionary where key is z, value is i
                df_correction.append(row)

        df_correction = pd.DataFrame(df_correction)

        # follows similar logic as feature extraction task
        if not df_correction.empty:
            correction_table = format_roi_table([df_correction])
            # Raise warnings if any values of anndata X matrix are less than or equal to a low_threshold value, or
            # greater than a high_threshold value.
            check_zillum_correction_table(
                correction_table,
                low_threshold=low_bound_for_correction,
                high_threshold=1.0,
            )
        else:
            # Create empty anndata table
            correction_table = ad.AnnData()

        # Write to zarr group
        image_group = zarr.group(f"{zarr_url}")
        table_attrs = {
            "type": "feature_table",
            "fractal_table_version": "1",
            "region": dict(path=f"../labels/{label_name}"),
            "instance_key": "label",
        }

        write_name = f"{output_table_name}_{channel_id}"
        write_table(
            image_group,
            write_name,
            correction_table,
            overwrite=True,
            table_attrs=table_attrs,
        )

        logger.info(
            f"Successfully processed {object_count} out of {total_label_count} labels for channel {channel}."
        )

        logger.info(
            f"Saved z-illumination correction matrix as Anndata table {write_name}"
        )

        logger.info(f"Saved quality control plots per object in {filepath}.")

    logger.info("End calculate_z_illumination_correction task.")

    return {}


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=calculate_z_illumination_correction,
        logger_name=logger.name,
    )
