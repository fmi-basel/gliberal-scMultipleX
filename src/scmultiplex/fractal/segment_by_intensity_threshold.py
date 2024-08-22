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
from typing import Any

import anndata as ad
import dask.array as da
import numpy as np
import zarr
from fractal_tasks_core.channels import (
    ChannelInputModel,
    ChannelNotFoundError,
    OmeroChannel,
    get_channel_from_image_zarr,
)
from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.pyramids import build_pyramid
from fractal_tasks_core.roi import (
    check_valid_ROI_indices,
    convert_indices_to_regions,
    convert_ROI_table_to_indices,
    get_overlapping_pairs_3D,
    load_region,
)
from fractal_tasks_core.tables import write_table
from fractal_tasks_core.tasks.io_models import InitArgsRegistrationConsensus
from pydantic import validate_call

from scmultiplex.fractal.fractal_helper_functions import (
    format_roi_table,
    initialize_new_label,
    save_new_label_with_overlap,
)
from scmultiplex.meshing.LabelFusionFunctions import (
    linear_z_correction,
    run_thresholding,
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
    channel_2: ChannelInputModel,
    background_channel_2: int = 400,
    intensity_threshold: int = 1500,
    gaussian_sigma_raw_image: float = 20,
    gaussian_sigma_threshold_image: float = 20,
    small_objects_diameter: float = 30,
    canny_threshold: float = 0.2,
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
        channel_2: Channel of second raw image to be combined with channel 1 image. Requires either
            `wavelength_id` (e.g. `A02_C02`) or `label` (e.g. `BCAT`).
        background_channel_2: Pixel intensity value of background to subtract from channel 2 raw image.
        intensity_threshold: Integer that specifies threshold intensity value to binarize image. Intensities below this
            value will be set to 0, intensities above are set to 1. The specified value should correspond to intensity
            range of raw image (e.g. for 16-bit images, 0-65535). Recommended threshold value is above image background
            level and below dimmest regions of image, particularly at deeper z-depth.
        gaussian_sigma_raw_image: Float that specifies sigma (standard deviation, in pixels)
            for 3D Gaussian kernel used for blurring of raw intensity image prior to thresholding and edge detection.
            Higher values correspond to more blurring that reduce holes in thresholded image. Recommended range 10-30.
        gaussian_sigma_threshold_image: Float that specifies sigma (standard deviation, in pixels)
            for 2D Gaussian kernel used for blurring each z-slice of thresholded binary image prior to edge detection.
            Higher values correspond to more blurring and smoother surface edges. Recommended range 10-30.
        small_objects_diameter: Float that specifies the approximate diameter, in pixels and at level=0, of debris in
            the image. This value is used to filter out small objects using skimage.morphology.remove_small_objects.
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

    ##############
    # Load segmentation image  ###
    ##############

    # Lazily load zarr array for reference cycle
    # load well image as dask array e.g. for nuclear segmentation
    label_dask = da.from_zarr(f"{zarr_url}/labels/{label_name}/{level}")

    # Read ROIs of objects
    roi_adata = ad.read_zarr(f"{zarr_url}/tables/{roi_table}")

    # Read Zarr metadata
    label_ngffmeta = load_NgffImageMeta(f"{zarr_url}/labels/{label_name}")
    label_xycoars = (
        label_ngffmeta.coarsening_xy
    )  # need to know when building new pyramids
    label_pixmeta = label_ngffmeta.get_pixel_sizes_zyx(level=level)

    # Create list of indices for 3D ROIs spanning the entire Z direction
    # Note that this ROI list is generated based on the input ROI table; if the input ROI table is for the group_by
    # objects, then label regions will be loaded based on the group_by ROIs
    roi_idlist = convert_ROI_table_to_indices(
        roi_adata,
        level=level,
        coarsening_xy=label_xycoars,
        full_res_pxl_sizes_zyx=label_pixmeta,
    )

    check_valid_ROI_indices(roi_idlist, roi_table)

    if len(roi_idlist) == 0:
        logger.warning("Well contains no objects")

    ##############
    # Load Channel images  ###
    ##############

    # Find channel index for channel 1
    try:
        tmp_channel1: OmeroChannel = get_channel_from_image_zarr(
            image_zarr_path=f"{zarr_url}",
            wavelength_id=channel_1.wavelength_id,
            label=channel_1.label,
        )
    except ChannelNotFoundError as e:
        logger.warning(
            "Channel not found, exit from the task.\n" f"Original error: {str(e)}"
        )
        return {}

    channel_1_id = tmp_channel1.index

    # Find channel index for channel 2
    try:
        tmp_channel2: OmeroChannel = get_channel_from_image_zarr(
            image_zarr_path=f"{zarr_url}",
            wavelength_id=channel_2.wavelength_id,
            label=channel_2.label,
        )
    except ChannelNotFoundError as e:
        logger.warning(
            "Channel not found, exit from the task.\n" f"Original error: {str(e)}"
        )
        return {}

    channel_2_id = tmp_channel2.index

    # Load channel data
    ch1_dask_raw = da.from_zarr(f"{zarr_url}/{level}")[channel_1_id]
    ch2_dask_raw = da.from_zarr(f"{zarr_url}/{level}")[channel_2_id]

    # Read Zarr metadata
    ngffmeta_raw = load_NgffImageMeta(f"{zarr_url}")
    xycoars_raw = ngffmeta_raw.coarsening_xy
    pixmeta_raw = ngffmeta_raw.get_pixel_sizes_zyx(level=level)

    ch1_idlist_raw = convert_ROI_table_to_indices(
        roi_adata,
        level=level,
        coarsening_xy=xycoars_raw,
        full_res_pxl_sizes_zyx=pixmeta_raw,
    )

    ch2_idlist_raw = convert_ROI_table_to_indices(
        roi_adata,
        level=level,
        coarsening_xy=xycoars_raw,
        full_res_pxl_sizes_zyx=pixmeta_raw,
    )

    check_valid_ROI_indices(ch1_idlist_raw, roi_table)
    check_valid_ROI_indices(ch2_idlist_raw, roi_table)

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

        # Load label image of label_name object as numpy array
        seg = load_region(
            data_zyx=label_dask,
            region=convert_indices_to_regions(roi_idlist[row_int]),
            compute=compute,
        )

        # Load channel 1 raw image for object
        ch1_raw = load_region(
            data_zyx=ch1_dask_raw,
            region=convert_indices_to_regions(ch1_idlist_raw[row_int]),
            compute=compute,
        )

        # Load channel 2 raw image for object
        ch2_raw = load_region(
            data_zyx=ch2_dask_raw,
            region=convert_indices_to_regions(ch2_idlist_raw[row_int]),
            compute=compute,
        )

        if seg.shape != ch1_raw.shape:
            raise ValueError("Shape of label and raw images must match.")

        # Check that label exists in object
        if float(label_str) not in seg:
            raise ValueError(
                f"Object ID {label_str} does not exist in loaded segmentation image. Does input ROI "
                f"table match label map?"
            )
        # Select label that corresponds to current object, set all other objects to 0
        seg[seg != float(label_str)] = 0
        # Binarize object segmentation image
        seg[seg > 0] = 1

        ch1_raw[ch1_raw <= background_channel_1] = 0
        ch1_raw[
            ch1_raw > 0
        ] -= background_channel_1  # will never have negative values this way

        ch2_raw[ch2_raw <= background_channel_2] = 0
        ch2_raw[
            ch2_raw > 0
        ] -= background_channel_2  # will never have negative values this way

        # Combine raw images
        # TODO: make second channel optional, can also use only 1 image
        # TODO: weighed sum of channel images (multiply each channel image by scalar)

        combo = 0.5 * ch1_raw + 0.5 * ch2_raw  # temporary: take average
        # TODO: correct check here that values above 65535 are not clipped; generalize to different input types
        combo[combo > 65535] = 65535

        if linear_z_illumination_correction:
            combo = linear_z_correction(combo, start_z_slice, m_slope)

        # TODO: consider using https://github.com/seung-lab/fill_voids to fill luman holes
        # TODO: account for z-decay of intensity
        # TODO: update Zenodo test dataset so that org seg matches raw image level
        seg3d, padded_zslice_count, roi_count = run_thresholding(
            combo,
            intensity_threshold,
            gaussian_sigma_raw_image,
            gaussian_sigma_threshold_image,
            small_objects_diameter,
            canny_threshold,
            pixmeta_raw,
            seg,
        )

        # Check whether is binary
        if np.amax(seg3d) not in [0, 1]:
            raise ValueError("Image not binary")

        if roi_count > 0:
            logger.info(
                f"Successfully calculated 3D label map for object label {label_str}."
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

        # Value of binary label image is set to value of current object here
        bbox_df = save_new_label_with_overlap(
            seg3d,
            label_str,
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
