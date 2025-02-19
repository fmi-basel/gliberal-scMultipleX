# Copyright (C) 2024 Friedrich Miescher Institute for Biomedical Research

##############################################################################
#                                                                            #
# Author: Nicole Repina              <nicole.repina@fmi.ch>                  #
#                                                                            #
##############################################################################


"""
Calculate full 3D object segmentation after 2D MIP-based segmentation using intensity thresholding of
raw intensity image(s).
"""
import logging
from typing import Any, Optional

import numpy as np
import zarr
from fractal_tasks_core.channels import ChannelInputModel
from fractal_tasks_core.pyramids import build_pyramid
from fractal_tasks_core.roi import (
    convert_indices_to_regions,
    get_overlapping_pairs_3D,
    load_region,
)
from fractal_tasks_core.tables import write_table
from fractal_tasks_core.tasks.io_models import InitArgsRegistrationConsensus
from pydantic import validate_call

from scmultiplex.fractal.fractal_helper_functions import (
    format_roi_table,
    initialize_new_label,
    load_channel_image,
    load_image_array,
    load_label_rois,
    load_seg_and_raw_region,
    save_new_label_and_bbox_df,
)
from scmultiplex.meshing.LabelFusionFunctions import (
    linear_z_correction,
    rescale_channel_image,
    run_thresholding,
    select_label,
)

logger = logging.getLogger(__name__)


@validate_call
def segment_by_intensity_threshold(
    *,
    # Fractal arguments
    zarr_url: str,
    init_args: InitArgsRegistrationConsensus,
    # Task-specific arguments
    label_name: str = "org",
    roi_table: str = "org_ROI_table",
    output_label_name: str = "org3d",
    channel_1: ChannelInputModel,
    background_channel_1: int = 800,
    maximum_channel_1: int,
    weight_channel_1: float = 0.5,
    combine_with_channel_2: bool = False,
    channel_2: Optional[ChannelInputModel] = None,
    background_channel_2: Optional[int] = None,
    maximum_channel_2: Optional[int] = None,
    weight_channel_2: float = 0.5,
    otsu_threshold: bool = True,
    otsu_weight: float = 1.0,
    intensity_threshold: int = -1,
    gaussian_sigma_raw_image: float = 30,
    gaussian_sigma_threshold_image: float = 20,
    small_objects_diameter: float = 30,
    expand_by_pixels: int = 20,
    canny_threshold: float = 0.4,
    linear_z_illumination_correction: bool = False,
    start_z_slice: int = 40,
    m_slope: float = 0.015,
) -> dict[str, Any]:
    """
    Calculate full 3D object segmentation after 2D MIP-based segmentation using intensity thresholding of
    raw intensity image(s).

    This task consists of 3 parts:

    1. Load the intensity images for selected channels using MIP-based segmentation ROIs.
    2. Generate 3D mask based on simple thresholding of the combined channel images. The thresholded image is
        smoothened using gaussian blurring followed by Canny edge detection. Optional z-illumination correctopn
        is applied on the fly. The MIP-based segmentation is used to mask
        the resulting label image to roughly exclude any neighboring organoids and debris. To further exclude
        neighboring organoids and debris, the largest connected component is selected as the final label image.
    3. Output: save the (1) new label image and (2) new masking ROI table in the selected zarr url.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            Refers to the zarr_url of the reference acquisition.
            (standard argument for Fractal tasks, managed by Fractal server).
        init_args: Intialization arguments provided by
            `init_group_by_well_for_multiplexing`. It contains the
            zarr_url_list listing all the zarr_urls in the same well as the
            zarr_url of the reference acquisition that are being processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        label_name: Label name of segmentation (usually based on 2D MIP) that identifies objects in image.
        roi_table: Name of the ROI table that corresponds to label_name. This table is used to iterate over
            objects and load object regions.
        output_label_name: Desired name for new output label. The corresponding ROI
            table will be saved as {output_label_name}_ROI_table.
        channel_1: Channel of raw image used for thresholding. Requires either
            `wavelength_id` (e.g. `A01_C01`) or `label` (e.g. `DAPI`).
        background_channel_1: Pixel intensity value of background to subtract from channel 1 raw image.
        maximum_channel_1: Maximum pixel intensity value that channel 1 image is rescaled to.
        weight_channel_1: Float specifying weight of channel 1 image. Channels are combined as
            (weight_channel_1 * ch1_raw) + (weight_channel_2 * ch2_raw). When both weights are 0.5, channels
            are averaged. If no second channel is provided, this parameter is ignored.
        combine_with_channel_2: if True, a second channel can be added. The Channel 1 and 2 images are combined using
            weights specified with weight_channel_1 and weight_channel_2, and thresholding is performed using this
            combined image.
        channel_2: Channel of second raw image to be combined with channel 1 image. Requires either
            `wavelength_id` (e.g. `A02_C02`) or `label` (e.g. `BCAT`).
        background_channel_2: Pixel intensity value of background to subtract from channel 2 raw image.
        maximum_channel_2: Maximum pixel intensity value that channel 1 image is rescaled to.
        weight_channel_2: Float specifying weight of channel 2 image. Channels are combined as
            (weight_channel_1 * ch1_raw) + (weight_channel_2 * ch2_raw)
        otsu_threshold: if True, the threshold for each region is calculated with the Otsu method. This threshold
            method is more robust to intensity variation between objects compared to intensity_threshold.
        otsu_weight: Scale calculated Otsu threhsold by this value. Values lower than 1 (e.g. 0.9) reduce Otsu
            threshold (e.g. by 10 %) to include lower-intensity pixels in thresholding.
        intensity_threshold: Integer that specifies threshold intensity value to binarize image.
            Must be supplied if Otsu thresholding is not used. Intensities below this
            value will be set to 0, intensities above are set to 1. The specified value should correspond to intensity
            range of raw image (e.g. for 16-bit images, 0-65535). Recommended threshold value is above image background
            level and below dimmest regions of image, particularly at deeper z-depth.
        gaussian_sigma_raw_image: Float that specifies sigma (standard deviation, in pixels)
            for 3D Gaussian kernel used for blurring of raw intensity image prior to thresholding and edge detection.
            Higher values correspond to more blurring that reduce holes in thresholded image. Recommended range 10-40.
        gaussian_sigma_threshold_image: Float that specifies sigma (standard deviation, in pixels)
            for 2D Gaussian kernel used for blurring each z-slice of thresholded binary image prior to edge detection.
            Higher values correspond to more blurring and smoother surface edges. Recommended range 10-30.
        small_objects_diameter: Float that specifies the approximate diameter, in pixels and at level=0, of debris in
            the image. This value is used to filter out small objects using skimage.morphology.remove_small_objects.
        expand_by_pixels: Expand initial threshold mask by this number of pixels and fill holes. Mask is subsequently
            dilated and returned to original size. This step serves to fill holes in dim regions. Higher values lead
            to more holes filled, but neighboring objects or debris may become fused.
        canny_threshold: Float in range [0,1]. Image values below this threshold are set to 0 after
            Gaussian blur using gaus_sigma_thresh_img. Higher threshold values result in tighter fit of edge mask
            to intensity image.
        linear_z_illumination_correction: Set to True if linear z illumination correction is desired. Iterate over
            z-slices to apply correction.
        start_z_slice: Z-slice number at which to begin to apply linear correction, e.g. slice 40 if
            image stack has 100 slices.
        m_slope: Slope factor of illumination correction. Higher values have more extreme correction. This value sets
            the multiplier for a given z-slice by formula m_slope * (i - start_z_slice) + 1, where i is the current
            z-slice in iterator.
    """

    logger.info(
        f"Running for {zarr_url=}. \n"
        f"Calculating 3D label map per {roi_table=} for "
        f"{label_name=}."
    )

    # Always use highest resolution label
    level = 0

    if otsu_threshold:
        threshold_type = "otsu"
        logger.info("Running thresholding with Otsu threshold")
    elif intensity_threshold > 0:
        threshold_type = "user-defined"
        logger.info(
            f"Running thresholding with user-defined threshold of {intensity_threshold}"
        )
    else:
        raise ValueError(
            "If Otsu threshold is not desired, user must provide non-negative "
            "intensity threshold value."
        )

    if maximum_channel_1 < background_channel_1:
        raise ValueError("Maximum value of image must be higher than image background.")

    if combine_with_channel_2 and maximum_channel_2 < background_channel_2:
        raise ValueError("Maximum value of image must be higher than image background.")

    ##############
    # Load segmentation image  ###
    ##############

    label_dask, roi_adata, roi_idlist, label_ngffmeta, label_pixmeta = load_label_rois(
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

    # Load channel 1 dask array and ID list
    ch1_dask_raw, ch1_idlist_raw = load_channel_image(
        channel_1,
        img_array,
        zarr_url,
        level,
        roi_table,
        roi_adata,
        xycoars_raw,
        pixmeta_raw,
    )

    # Optionally load channel 2 dask array and ID list
    if combine_with_channel_2:
        ch2_dask_raw, ch2_idlist_raw = load_channel_image(
            channel_2,
            img_array,
            zarr_url,
            level,
            roi_table,
            roi_adata,
            xycoars_raw,
            pixmeta_raw,
        )

    ##############
    # Initialize parameters to save the newly calculated label map  ###
    ##############
    output_roi_table_name = f"{output_label_name}_ROI_table"

    shape = label_dask.shape
    chunks = label_dask.chunksize

    new_label3d_array = initialize_new_label(
        zarr_url, shape, chunks, np.uint32, label_name, output_label_name, logger
    )

    logger.info(f"Mask will have shape {shape} and chunks {chunks}")

    # initialize new ROI table
    bbox_dataframe_list = []

    ##############
    # Iterate over objects and perform segmentation  ###
    ##############

    # Get labels to iterate over
    roi_labels = roi_adata.obs_vector("label")
    total_label_count = len(roi_labels)
    compute = True
    object_count = 0

    logger.info(
        f"Starting iteration over {total_label_count} detected objects in ROI table."
    )

    # For each object in input ROI table...
    for row in roi_adata.obs_names:
        row_int = int(row)
        label_str = roi_labels[row_int]

        seg, ch1_raw = load_seg_and_raw_region(
            label_dask, ch1_dask_raw, roi_idlist, ch1_idlist_raw, row_int, compute
        )

        if seg.shape != ch1_raw.shape:
            raise ValueError("Shape of label and raw images must match.")

        # Check that label exists in object
        if float(label_str) not in seg:
            raise ValueError(
                f"Object ID {label_str} does not exist in loaded segmentation image. Does input ROI "
                f"table match label map?"
            )

        seg = select_label(seg, label_str)
        ch1_raw_rescaled = rescale_channel_image(
            ch1_raw, background_channel_1, maximum_channel_1
        )

        if combine_with_channel_2:
            # Load channel 2 raw image for object
            ch2_raw = load_region(
                data_zyx=ch2_dask_raw,
                region=convert_indices_to_regions(ch2_idlist_raw[row_int]),
                compute=compute,
            )

            ch2_raw_rescaled = rescale_channel_image(
                ch2_raw, background_channel_2, maximum_channel_2
            )

            image_to_segment = (weight_channel_1 * ch1_raw_rescaled) + (
                weight_channel_2 * ch2_raw_rescaled
            )  # temporary: take average
            # TODO: correct check here that values above 65535 are not clipped; generalize to different input types
            image_to_segment[image_to_segment > 65535] = 65535

        else:
            image_to_segment = ch1_raw_rescaled

        if linear_z_illumination_correction:
            image_to_segment = linear_z_correction(
                image_to_segment, start_z_slice, m_slope
            )
            # TODO: correct check here that values above 65535 are not clipped; generalize to different input types
            image_to_segment[image_to_segment > 65535] = 65535

        # TODO: consider using https://github.com/seung-lab/fill_voids to fill luman holes

        seg3d, padded_zslice_count, roi_count, threshold = run_thresholding(
            image_to_segment,
            threshold_type,
            gaussian_sigma_raw_image,
            gaussian_sigma_threshold_image,
            small_objects_diameter,
            expand_by_pixels,
            canny_threshold,
            pixmeta_raw,
            seg,
            intensity_threshold,
            otsu_weight,
        )

        # Check whether is binary
        if np.amax(seg3d) not in [0, 1]:
            raise ValueError("Image not binary")

        if roi_count > 0:
            logger.info(
                f"Successfully calculated 3D label map for object label {label_str} using "
                f"threshold {np.round(threshold,1)}."
            )
            object_count += 1
            if roi_count > 1:
                logger.info(
                    f"Object {label_str} contains more than 1 component. "
                    f"Largest component selected as label mask."
                )
        else:
            logger.warning(
                f"Empty result for object label  {label_str}. No label calculated. "
                f"Is small_objects_diameter or intensity_threshold too high?"
            )

        if padded_zslice_count > 0:
            logger.info(
                f"Object {label_str} has non-zero pixels touching image border. Image processing "
                f"completed successfully, however consider reducing sigma "
                f"or increasing the canny_threshold to reduce risk of cropping shape edges."
            )

        ##############
        # Save labels and make ROI table ###
        ##############

        # Store labels as new label map in zarr
        # Note that pixels of overlap in the case where two labels are touching are overwritten by the last
        # written object. However, this should not change existing image borders from 2D MIP segmentation since the
        # new 3D label map is masked by the 2D MIP label.

        # Convert edge detection label image value to match object label id
        seg3d = seg3d * int(label_str)

        # Value of binary label image is set to value of current object here
        bbox_df = save_new_label_and_bbox_df(
            seg3d,
            new_label3d_array,
            zarr_url,
            output_label_name,
            convert_indices_to_regions(roi_idlist[row_int]),
            label_pixmeta,
            compute,
            roi_idlist,
            row_int,
        )
        bbox_dataframe_list.append(bbox_df)

        overlap_list = []
        for df in bbox_dataframe_list:
            overlap_list.extend(get_overlapping_pairs_3D(df, label_pixmeta))

        if len(overlap_list) > 0:
            logger.warning(f"{len(overlap_list)} bounding-box pairs overlap")

    # Starting from on-disk highest-resolution data, build and write to disk a pyramid of coarser levels
    build_pyramid(
        zarrurl=f"{zarr_url}/labels/{output_label_name}",
        overwrite=True,
        num_levels=label_ngffmeta.num_levels,
        coarsening_xy=label_ngffmeta.coarsening_xy,
        chunksize=chunks,
        aggregation_function=np.max,
    )

    logger.info(
        f"Built a pyramid for the {zarr_url}/labels/{output_label_name} label image"
    )

    bbox_table = format_roi_table(bbox_dataframe_list)
    # Write to zarr group
    logger.info(
        f"Writing new bounding-box ROI table to {zarr_url}/tables/{output_roi_table_name}"
    )

    table_attrs = {
        "type": "ngff:region_table",
        "region": {"path": f"../labels/{output_label_name}"},
        "instance_key": "label",
    }

    write_table(
        zarr.group(zarr_url),
        output_roi_table_name,
        bbox_table,
        overwrite=True,
        table_attrs=table_attrs,
    )

    logger.info(
        f"Successfully processed {object_count} out of {total_label_count} labels."
    )
    logger.info(
        f"End segment_by_intensity_threshold task for {zarr_url}/labels/{label_name}"
    )

    return {}


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=segment_by_intensity_threshold,
        logger_name=logger.name,
    )
