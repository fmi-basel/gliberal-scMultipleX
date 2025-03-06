# Copyright (C) 2024 Friedrich Miescher Institute for Biomedical Research

##############################################################################
#                                                                            #
# Author: Nicole Repina              <nicole.repina@fmi.ch>                  #
#                                                                            #
##############################################################################

import dask.array as da
import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion, binary_fill_holes
from skimage.draw import polygon
from skimage.exposure import rescale_intensity
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import find_contours, label, regionprops, regionprops_table
from skimage.morphology import disk, remove_small_objects
from skimage.segmentation import expand_labels

from scmultiplex.features.FeatureFunctions import mesh_equivalent_diameter
from scmultiplex.linking.NucleiLinkingFunctions import remove_labels
from scmultiplex.meshing.FilterFunctions import (
    add_xy_pad,
    calculate_mean_volume,
    filter_small_sizes_per_round,
    load_border_values,
    remove_border,
    remove_xy_pad,
)


def get_expandby_pixels(seg, expandby_factor):
    """
    Determine the expansion amount (in pixels) based on the mean diameter of 3D nuclei or cells in object
    Returns integer
    """
    size_mean = calculate_mean_volume(seg)
    try:
        expandby_pix = int(round(expandby_factor * mesh_equivalent_diameter(size_mean)))
    # Catch error in case no objects in volume
    except ValueError:
        expandby_pix = 0
    return expandby_pix


def expand_2d_labels(seg, expandby_pix):
    """
    Expand 2D numpy array by expandby_pix pixels without overlapping
    expandby_pix is integer, specifies number of pixels to expand by
    Shape of expanded object is equal to input
    """

    # Initialize empty array
    if len(seg.shape) == 2:
        seg_expanded = expand_labels(seg, expandby_pix)

    elif len(seg.shape) == 3:
        seg_expanded = np.zeros_like(seg)

        # Iterate over each zslice in image and expand
        for i, zslice in enumerate(seg):
            seg_expanded[i, :, :] = expand_labels(zslice, expandby_pix)
    else:
        raise ValueError("Input segmentation must be 2d or 3d array.")

    return seg_expanded


def fuse_labels(seg, expandby_factor):
    """
    Expand labels in 3D image by 2D z-slice based on the mean volume, fuse and fill holes,
    and erode back to original size.
    """
    # Initialize empty array
    seg_binary = np.zeros_like(seg)

    # Determine the expansion amount (in pixels) based on the mean diameter of 3D nuclei or cells in object
    expandby_pix = get_expandby_pixels(seg, expandby_factor)

    # Iterate over each zslice in image
    for i, zslice in enumerate(seg):
        # Pad image
        zslice = np.pad(zslice, pad_width=expandby_pix)
        # Expand labels in x,y
        zslice = expand_labels(zslice, expandby_pix)
        # Fill holes and binarize area (return boolean)
        zslice = binary_fill_holes(zslice)
        # Revert the expansion (i.e. erode down to original size)
        # Erode by half of the expanded pixels since disk(1) has a radius of 1, i.e. diameter of 2
        eroded = binary_erosion(
            zslice, disk(1), iterations=expandby_pix
        )  # returns binary image, max val 1
        # remove padding
        seg_binary[i, :, :] = remove_border(eroded, pad_width=expandby_pix)

    return seg_binary, expandby_pix


def anisotropic_gaussian_blur(seg_binary, sigma, pixmeta, convert_to_8bit=True):
    """
    Perform gaussian blur of binary 3D image with anisotropic sigma. Sigma anisotropy calculated from pixel spacing.
    """
    # Scale sigma to match pixel anisotropy
    spacing = (
        np.array(pixmeta) / pixmeta[1]
    )  # (z, y, x) where x,y is normalized to 1 e.g. (3, 1, 1)
    anisotropic_sigma = tuple(sigma * (spacing[1] / x) for x in spacing)

    # Perform 3D gaussian blur
    # Output image has value range 0-1 (not always max 1)
    if convert_to_8bit:
        # Convert binary segmentation into 8-bit image
        seg_fill_8bit = (seg_binary * 255).astype(np.uint8)
        blurred = gaussian(seg_fill_8bit, sigma=anisotropic_sigma, preserve_range=False)
    else:
        blurred = gaussian(seg_binary, sigma=anisotropic_sigma, preserve_range=True)

    return blurred, anisotropic_sigma


def find_edges(cleaned, contour_value, min_size, segment_lumen=False):
    """
    Find edges of input 3D image by z-slice
    """
    # Initialize empty array
    outer_stack = np.zeros_like(cleaned)

    if segment_lumen:
        lumen_stack = np.zeros_like(cleaned)

    # Values below Canny threshold are set to 0. This is similar to setting a contour value.
    cleaned[cleaned < contour_value] = 0

    # Count the number of zslices that require padding
    padded_zslice_count = 0

    # Iterate over each zslice in image
    for i, zslice in enumerate(cleaned):

        padded = False
        lumen_found = False

        # If zslice is empty, skip the zslice
        if np.count_nonzero(zslice) == 0:
            continue

        # Get pixel values at borders of image
        image_border = load_border_values(zslice)

        # If any values are non-zero, object is touching image edge and thus requires zero-padding
        # This ensures that Canny filter generates a closed surface
        if np.any(image_border):
            zslice = np.pad(zslice, 1)
            padded_zslice_count += 1
            padded = True

        outer = np.zeros_like(zslice)

        contours = find_contours(zslice, level=contour_value, fully_connected="high")
        # if any contours are detected...
        if contours:
            # Sort contours by length (assuming the two longest ones are the ones we need)
            contours = sorted(contours, key=len, reverse=True)

            # assume largest contour is contour of outer epithelium
            if len(contours) > 0:
                outer_contour = contours[0]
                # identify pixels belonging to inside of contour
                rr, cc = polygon(
                    outer_contour[:, 0], outer_contour[:, 1], shape=zslice.shape
                )
                outer[rr, cc] = 1  # Set pixels inside the polygon to 1

            # and that second-largest contour is lumen
            if segment_lumen and len(contours) > 1:
                lumen = np.zeros_like(zslice)
                inner_contour = contours[1]
                # identify pixels belonging to inside of contour
                rr, cc = polygon(
                    inner_contour[:, 0], inner_contour[:, 1], shape=zslice.shape
                )
                lumen[rr, cc] = 1  # Set pixels inside the polygon to 1
                lumen_found = True

        # Remove padding
        if padded:
            outer = remove_border(outer)
            if lumen_found:
                lumen = remove_border(lumen)

        outer_stack[i, :, :] = outer

        if lumen_found:
            lumen_stack[i, :, :] = lumen

    # Convert to 8-bit image
    outer_stack = (outer_stack * 255).astype(np.uint8)
    # Filter out small objects that are smaller than radius of expansion and convert to labelmap
    outer_stack = label(remove_small_objects(outer_stack, min_size))

    if segment_lumen:
        lumen_stack = (lumen_stack * 255).astype(np.uint8)
        lumen_stack = label(remove_small_objects(lumen_stack, min_size / 20))
        return outer_stack, lumen_stack, padded_zslice_count

    return outer_stack, padded_zslice_count


def filter_by_volume(seg, volume_filter_threshold):
    """
    Remove segmentations from label image that have a volume less than specified volume threshold,
    which is a fraction of the median object size in image
    """
    # Run regionprops to extract centroids and volume of each label
    seg_props = regionprops_table(
        label_image=seg, properties=("label", "centroid", "area")
    )  # zyx

    # Convert to pandas, then numpy
    # Output column order must be: ["label", "x_centroid", "y_centroid", "z_centroid", "volume"]
    seg_props = (
        pd.DataFrame(
            seg_props,
            columns=["label", "centroid-2", "centroid-1", "centroid-0", "area"],
        )
    ).to_numpy()

    # Discard segmentations that have a volume less than fraction volume threshold
    (
        seg_props,
        segids_toremove,
        removed_size_mean,
        size_mean,
        volume_cutoff,
    ) = filter_small_sizes_per_round(
        seg_props, column=-1, threshold=volume_filter_threshold
    )

    segids_toremove = list(segids_toremove)  # list of float64

    # Relabel input image to remove identified IDs
    datatype = seg.dtype
    seg_filtered = remove_labels(seg, segids_toremove, datatype)

    return seg_filtered, segids_toremove, removed_size_mean, size_mean, volume_cutoff


def linear_z_correction(raw_image, start_thresh, m):
    corrected_image = np.zeros_like(raw_image)
    for i, zslice in enumerate(raw_image):
        if i > start_thresh:
            factor = m * (i - start_thresh) + 1
        else:
            factor = 1
        zslice = zslice * factor
        corrected_image[i] = zslice
    return corrected_image


def clean_binary_image(
    image_binary, sigma2d, small_objects_threshold, expandby_pix, fill_holes=True
):
    """
    Clean up an input binary image (0/1) by filling holes, expanding labels and filling holes,
    subsequent dilation, removing small objects (below small_objects_threshold),
    and performing gaussian blur (with sigma2d).
    Performed in 2D on each zslice of image.
    Return cleaned up binary (0-1) image, float 64, same shape as input array.
    """
    cleaned = np.zeros_like(image_binary, dtype=np.float64)

    iterations = int(round(expandby_pix / 2))

    # Iterate over each zslice in image
    for i, zslice in enumerate(image_binary):
        if fill_holes:
            zslice = binary_fill_holes(zslice)
        zslice = expand_labels(
            zslice, expandby_pix
        )  # expand mask to fuse labels and hill holes
        if fill_holes:
            zslice = binary_fill_holes(zslice)
        zslice = binary_erosion(
            zslice, disk(1), iterations=iterations
        )  # dilate mask to original size
        zslice = remove_small_objects(zslice, small_objects_threshold)
        zslice = (zslice * 255).astype(np.uint8)  # convert 0-255
        zslice = gaussian(zslice, sigma=sigma2d, preserve_range=False)  # output is 0-1
        cleaned[i, :, :] = zslice

    return cleaned


def threshold_image(image, intensity_threshold):
    # Generate binary mask
    # Simple intensity threshold; values above threshold set to 1, values below set to 0
    image_thresholded = np.where(image > intensity_threshold, 1, 0)
    return image_thresholded


def select_largest_component(label_image):
    """
    Select largest connected component of label image. Discard all smaller components.
    Return binary (0/1) image.
    """
    rois = regionprops(label_image)
    roi_count = len(rois)

    if roi_count > 1:
        label_with_largest_volume = 0
        largest_volume = 0
        for r in rois:
            rlabel = r.label
            rvolume = r.area
            if rvolume > largest_volume:
                largest_volume = rvolume
                label_with_largest_volume = rlabel

        label_image[label_image != label_with_largest_volume] = 0

    # binarize all images
    label_image[label_image > 0] = 1

    return label_image, roi_count


def run_label_fusion(
    seg,
    parent_seg,
    expandby_factor,
    sigma,
    pixmeta,
    canny_threshold,
    xy_padwidth,
    mask_by_parent=False,
):
    """
    Main function for running label fusion. Used in Surface Mesh Multiscale task to generate organoid label from
    single-cell segmentation.
    Note that padding is not removed unless mask_by_parent=True. This way meshes can be generated from nicely
    smoothened images. Pad must be removed after meshing is complete.
    """
    seg_binary, expandby_pix = fuse_labels(seg, expandby_factor)

    # Add padding equal to sigma in xy dimensions so that blur can spread beyond image edge.
    seg_binary = add_xy_pad(seg_binary, xy_padwidth)

    blurred, anisotropic_sigma = anisotropic_gaussian_blur(seg_binary, sigma, pixmeta)
    edges_canny, padded_zslice_count = find_edges(
        blurred, canny_threshold, expandby_pix
    )

    # Filter by organoid label image from 2D segmentation (converted to 3D)
    # This removes debris that is far from organoid (only in xy)
    # and at least partially removes touching neighboring organoids
    if mask_by_parent:
        # First remove padding so that image size matches parent mask
        edges_canny = remove_xy_pad(edges_canny, xy_padwidth)
        edges_canny = edges_canny * parent_seg

    # Discard small disconnected components
    contour, roi_count = select_largest_component(edges_canny)

    return (
        contour,
        expandby_pix,
        anisotropic_sigma,
        padded_zslice_count,
        roi_count,
    )


def run_expansion(seg, expandby, expansion_distance_image_based=False):
    """
    Main function for running label expansion, used in Expand Labels task.
    """
    if expansion_distance_image_based:
        expandby_pix = get_expandby_pixels(seg, expandby_factor=expandby)
    else:
        expandby_pix = expandby

    seg_expanded = expand_2d_labels(seg, expandby_pix)
    return seg_expanded, expandby_pix


def run_thresholding(
    raw_image,
    threshold_type,
    gaus_sigma_raw_img,
    gaus_sigma_thresh_img,
    small_objects_diameter,
    expand_by_pixel,
    canny_threshold,
    pixmeta_raw,
    seg,
    intensity_threshold,
    otsu_weight,
    segment_lumen,
):
    """
    Main function for running intensity-based thresholding.
    Used in Segment by Intensity Threshold Fractal task to generate 3D label map from intensity image.
    """
    if segment_lumen:
        fill_holes = False
    else:
        fill_holes = True

    # Mask raw image by input segmentation
    raw_image = raw_image * seg

    # Apply 3D gaussian blur to raw intensity image prior to thresholding
    blurred, anisotropic_sigma = anisotropic_gaussian_blur(
        raw_image, gaus_sigma_raw_img, pixmeta_raw, convert_to_8bit=False
    )

    if threshold_type == "otsu":
        threshold = threshold_otsu(image=blurred)
        threshold = otsu_weight * threshold
    elif threshold_type == "user-defined":
        threshold = intensity_threshold
    else:
        raise ValueError("Thresholding type not defined.")

    combo_binary = threshold_image(blurred, threshold)

    # Clean up binary mask by filling holes, removing small objects, and gaussian blur by z-slice
    small_objects_2dthreshold = np.pi * (small_objects_diameter / 2) ** 2
    cleaned = clean_binary_image(
        combo_binary,
        gaus_sigma_thresh_img,
        small_objects_2dthreshold,
        expand_by_pixel,
        fill_holes=fill_holes,
    )

    # Smoothen edges of mask via Canny edge detection
    small_objects_3dthreshold = (4 / 3) * np.pi * (small_objects_diameter / 2) ** 3
    if segment_lumen:

        contour, lumen, padded_zslice_count = find_edges(
            cleaned,
            canny_threshold,
            small_objects_3dthreshold,
            segment_lumen=segment_lumen,
        )

        # Filter by organoid label image from 2D segmentation (converted to 3D)
        # This removes debris that is far from organoid (only in xy)
        # and at least partially removes touching neighboring organoids
        contour = contour * seg
        lumen = lumen * seg

        # Discard small disconnected components
        contour, roi_count_contour = select_largest_component(contour)
        lumen, roi_count_lumen = select_largest_component(lumen)
        epithelium = contour - lumen

        return (
            contour,
            lumen,
            epithelium,
            padded_zslice_count,
            roi_count_contour,
            roi_count_lumen,
            threshold,
        )

    else:
        contour, padded_zslice_count = find_edges(
            cleaned,
            canny_threshold,
            small_objects_3dthreshold,
            segment_lumen=segment_lumen,
        )
        contour = contour * seg
        contour, roi_count_contour = select_largest_component(contour)

    return contour, padded_zslice_count, roi_count_contour, threshold


def rescale_channel_image(ch_raw, background_intensity, maximum_intensity):

    """Remove background and rescale intensity to a maximum value. Input is 3D numpy array image."""

    ch_raw[ch_raw <= background_intensity] = 0
    ch_raw[
        ch_raw > 0
    ] -= background_intensity  # will never have negative values this way

    ch_raw_rescaled = rescale_intensity(
        ch_raw, in_range=(0, maximum_intensity - background_intensity)
    )

    return ch_raw_rescaled


def select_label(seg, label_str):
    # Select label that corresponds to current object, set all other objects to 0
    seg[seg != float(label_str)] = 0
    # Binarize object segmentation image
    seg[seg > 0] = 1

    return seg


def simple_fuse_labels(label_dask, connectivity):
    dtype = label_dask.dtype
    dtype_max = np.iinfo(dtype).max

    # Set all values > 0 to True
    binary_array = label_dask > 0  # binary boolean array
    binary_numpy = binary_array.compute()

    # Default connectivity is maximum based on array dimensions
    if connectivity is None:
        connectivity = binary_numpy.ndim

    # Apply skimage label function
    fused_numpy, label_count = label(
        binary_numpy, background=0, return_num=True, connectivity=connectivity
    )

    if label_count > dtype_max:
        raise ValueError(
            f"Number of identified labels {label_count} exceeds {dtype} maximum of {dtype_max}."
        )

    # Convert back to dask to save on disk with same chunk sizes and dtype as input label map
    fused_dask = da.from_array(fused_numpy, chunks=label_dask.chunksize).astype(dtype)

    return fused_numpy, fused_dask, label_count, connectivity
