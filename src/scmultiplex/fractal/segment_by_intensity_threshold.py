# Copyright (C) 2024 Friedrich Miescher Institute for Biomedical Research

##############################################################################
#                                                                            #
# Author: Nicole Repina              <nicole.repina@fmi.ch>                  #
#                                                                            #
##############################################################################

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
from pydantic import validate_call

from scmultiplex.fractal.fractal_helper_functions import (
    format_roi_table,
    get_zattrs,
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
    init_args: dict,
    # Task-specific arguments
    label_name: str = "org",
    new_label_name: str = "org3d",
    roi_table: str = "org_ROI_table",
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
    contour_value_outer: float = 0.8,
    linear_z_illumination_correction: bool = False,
    start_z_slice: int = 40,
    m_slope: float = 0.015,
    segment_lumen: bool = False,
    contour_value_inner: float = 0.8,
) -> dict[str, Any]:
    """
    Calculate full 3D object segmentation after 2D MIP-based segmentation using intensity thresholding of
    raw intensity image(s).

    Assumes input is a masking roi table, i.e. that segmentation has been performed. Task does
    not work on roi tables e.g. well_ROI_table.

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
        init_args: Init arguments for Fractal server.
        label_name: Label name of segmentation (usually based on 2D MIP) that identifies objects in image.
        new_label_name: Desired name for new output label. The corresponding ROI
            table will be saved as {output_label_name}_ROI_table.
        roi_table: Name of the ROI table that corresponds to label_name. This table is used to iterate over
            objects and load object regions.
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
        contour_value_outer: Float in range [0,1]. This is the value used to draw contour line around object.
            Higher values result in tighter fit of edge mask to intensity image.
        linear_z_illumination_correction: Set to True if linear z illumination correction is desired. Iterate over
            z-slices to apply correction.
        start_z_slice: Z-slice number at which to begin to apply linear correction, e.g. slice 40 if
            image stack has 100 slices.
        m_slope: Slope factor of illumination correction. Higher values have more extreme correction. This value sets
            the multiplier for a given z-slice by formula m_slope * (i - start_z_slice) + 1, where i is the current
            z-slice in iterator.
        segment_lumen: if True, lumen (assumed to be negative space in object) will also be segmented. In this case,
            three label maps are output: outer contour (epithelial surface) with holes filled, inner contour (lumen),
            and the epithelial mask (difference between outer and inner regions).
        contour_value_inner: Float in range [0,1]. This is the value used to draw contour line around lumen of object.
            Higher values result in tighter fit of edge mask to intensity image.
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
    roi_attrs = get_zattrs(f"{zarr_url}/tables/{roi_table}")

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
    # Initialize parameters to save the newly calculated label map and ROI tables  ###
    ##############
    if segment_lumen:
        output_label_names = [
            f"{new_label_name}_outer",
            f"{new_label_name}_inner",
            f"{new_label_name}_diff",
        ]
        output_roi_names = [
            f"{new_label_name}_outer_ROI_table",
            f"{new_label_name}_inner_ROI_table",
            f"{new_label_name}_diff_ROI_table",
        ]
        bbox_df_lists = [[], [], []]

    else:
        output_label_names = [f"{new_label_name}_outer"]
        output_roi_names = [f"{new_label_name}_outer_ROI_table"]
        bbox_df_lists = [[]]

    shape = label_dask.shape
    chunks = label_dask.chunksize
    dtype = ch1_dask_raw.dtype
    dtype_max = np.iinfo(dtype).max

    new_label3d_arrays = []
    for i, name in enumerate(output_label_names):
        new_label3d_arrays.append(
            initialize_new_label(
                zarr_url, shape, chunks, np.uint32, label_name, name, logger
            )
        )

    logger.info(f"Mask will have shape {shape} and chunks {chunks}")

    ##############
    # Iterate over objects and perform segmentation  ###
    ##############

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
    object_count = 0

    logger.info(
        f"Starting iteration over {total_label_count} detected objects in ROI table."
    )

    # For each object in input ROI table...
    for i, obsname in enumerate(roi_adata.obs_names):
        label_str = roi_labels[i]

        seg, ch1_raw = load_seg_and_raw_region(
            label_dask, ch1_dask_raw, roi_idlist, ch1_idlist_raw, i, compute
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
                region=convert_indices_to_regions(ch2_idlist_raw[i]),
                compute=compute,
            )

            ch2_raw_rescaled = rescale_channel_image(
                ch2_raw, background_channel_2, maximum_channel_2
            )

            image_to_segment = (weight_channel_1 * ch1_raw_rescaled) + (
                weight_channel_2 * ch2_raw_rescaled
            )  # temporary: take average

            # Array is automatically upcasted if values higher than dtype_max is generated, here catch any overflow
            if np.amax(image_to_segment) > dtype_max:
                logger.warning(
                    f"Correction generated intensity values beyond the max range of input data type. "
                    f"These values have been clipped to {dtype_max=}"
                )
                # Correct clipped values; values above dtype_max are set to dtype_max
                image_to_segment = np.clip(image_to_segment, None, dtype_max)

        else:
            image_to_segment = ch1_raw_rescaled

        if linear_z_illumination_correction:
            image_to_segment = linear_z_correction(
                image_to_segment, start_z_slice, m_slope
            )

            # Array is automatically upcasted if values higher than dtype_max is generated, here catch any overflow
            if np.amax(image_to_segment) > dtype_max:
                logger.warning(
                    f"Correction generated intensity values beyond the max range of input data type. "
                    f"These values have been clipped to {dtype_max=}"
                )
                # Correct clipped values; values above dtype_max are set to dtype_max
                image_to_segment = np.clip(image_to_segment, None, dtype_max)

        # TODO: consider using https://github.com/seung-lab/fill_voids to fill luman holes

        result = run_thresholding(
            image_to_segment,
            threshold_type,
            gaussian_sigma_raw_image,
            gaussian_sigma_threshold_image,
            small_objects_diameter,
            expand_by_pixels,
            contour_value_outer,
            contour_value_inner,
            pixmeta_raw,
            seg,
            intensity_threshold,
            otsu_weight,
            segment_lumen=segment_lumen,
        )

        if segment_lumen:
            (
                contour,
                lumen,
                epithelium,
                padded_zslice_count,
                roi_count_contour,
                roi_count_lumen,
                threshold,
            ) = result
            labels_to_save = [contour, lumen, epithelium]
        else:
            contour, padded_zslice_count, roi_count_contour, threshold = result
            labels_to_save = [contour]

        # Check whether is binary
        if np.amax(contour) not in [0, 1]:
            raise ValueError("Image not binary")

        if roi_count_contour > 0:
            logger.info(
                f"Successfully calculated 3D label map for object label {label_str} using "
                f"threshold {np.round(threshold,1)}."
            )
            object_count += 1
            if roi_count_contour > 1:
                logger.info(
                    f"Object {label_str} contains more than 1 component. "
                    f"Largest component selected as label mask."
                )
        else:
            logger.warning(
                f"Empty result for object label  {label_str}. No label calculated. "
                f"Is small_objects_diameter or intensity_threshold too high?"
            )

        if segment_lumen:
            logger.info(f"Detected {roi_count_lumen} lumens in object.")

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

        for i, (out_label_name, np_img, array_on_disk, bbox_df_list) in enumerate(
            zip(output_label_names, labels_to_save, new_label3d_arrays, bbox_df_lists)
        ):
            # Convert edge detection label image value to match object label id
            np_img = np_img * int(label_str)

            # Value of binary label image is set to value of current object here
            bbox_df = save_new_label_and_bbox_df(
                np_img,
                array_on_disk,
                zarr_url,
                out_label_name,
                convert_indices_to_regions(roi_idlist[i]),
                label_pixmeta,
                compute,
                roi_idlist,
                i,
            )

            bbox_df_list.append(bbox_df)

            overlap_list = []
            for df in bbox_df_list:
                overlap_list.extend(get_overlapping_pairs_3D(df, label_pixmeta))

            if len(overlap_list) > 0:
                logger.warning(
                    f"{len(overlap_list)} bounding-box pairs overlap for label {out_label_name}"
                )

    for i, (out_label_name, out_roi_name, bbox_df_list) in enumerate(
        zip(output_label_names, output_roi_names, bbox_df_lists)
    ):

        # Starting from on-disk highest-resolution data, build and write to disk a pyramid of coarser levels
        build_pyramid(
            zarrurl=f"{zarr_url}/labels/{out_label_name}",
            overwrite=True,
            num_levels=label_ngffmeta.num_levels,
            coarsening_xy=label_ngffmeta.coarsening_xy,
            chunksize=chunks,
            aggregation_function=np.max,
        )

        logger.info(
            f"Built a pyramid for the {zarr_url}/labels/{out_label_name} label image"
        )

        bbox_table = format_roi_table(bbox_df_list)
        # Write to zarr group
        logger.info(
            f"Writing new bounding-box ROI table to {zarr_url}/tables/{out_roi_name}"
        )

        table_attrs = {
            "type": "ngff:region_table",
            "region": {"path": f"../labels/{out_label_name}"},
            "instance_key": "label",
        }

        write_table(
            zarr.group(zarr_url),
            out_roi_name,
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
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=segment_by_intensity_threshold,
        logger_name=logger.name,
    )
