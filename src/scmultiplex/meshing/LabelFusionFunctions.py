# Copyright (C) 2024 Friedrich Miescher Institute for Biomedical Research

##############################################################################
#                                                                            #
# Author: Nicole Repina              <nicole.repina@fmi.ch>                  #
#                                                                            #
##############################################################################

import numpy as np
from scipy.ndimage import binary_erosion, binary_fill_holes
from skimage.feature import canny
from skimage.filters import gaussian
from skimage.measure import label, regionprops
from skimage.morphology import disk, remove_small_objects
from skimage.segmentation import expand_labels

from scmultiplex.features.FeatureFunctions import mesh_equivalent_diameter
from scmultiplex.meshing.FilterFunctions import (
    calculate_mean_volume,
    load_border_values,
    remove_border,
)


def fuse_labels(seg, expandby_factor):
    """
    Expand labels in 3D image by 2D z-slice based on the mean volume, fuse and fill holes,
    and erode back to original size.
    """
    # Initialize empty array
    seg_binary = np.zeros_like(seg)

    # Determine the expansion amount (in pixels) based on the mean diameter of 3D nuclei or cells in object
    size_mean = calculate_mean_volume(seg)
    expandby_pix = int(round(expandby_factor * mesh_equivalent_diameter(size_mean)))

    # Determine the number of iterations for binary erosions to reverse the expansion
    # The number of iterations is roughly half of the expansion, since erosion is
    # performed with a disk size of diameter 2
    iterations = int(round(expandby_pix / 2))

    # Iterate over each zslice in image
    for i, zslice in enumerate(seg):
        # Expand labels in x,y
        zslice = expand_labels(zslice, expandby_pix)
        # Fill holes and binarize area (return boolean)
        zslice = binary_fill_holes(zslice)
        # Revert the expansion (i.e. erode down to original size)
        # Erode by half of the expanded pixels since disk(1) has a radius of 1, i.e. diameter of 2
        seg_binary[i, :, :] = binary_erosion(
            zslice, disk(1), iterations=iterations
        )  # returns binary image, max val 1

    return seg_binary, expandby_pix, iterations


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


def find_edges(blurred, canny_threshold, iterations):
    """
    Find edges of input 3D image by z-slice
    """
    # Initialize empty array
    edges_canny = np.zeros_like(blurred)

    # Values below Canny threshold are set to 0. This is similar to setting a contour value.
    blurred[blurred < canny_threshold] = 0

    # Count the number of zslices that require padding
    padded_zslice_count = 0

    # Iterate over each zslice in image
    for i, zslice in enumerate(blurred):

        padded = False

        # If zslice is empty, skip the zslice
        if np.count_nonzero(zslice) == 0:
            continue
        else:
            # Get pixel values at borders of image
            image_border = load_border_values(zslice)

            # If any values are non-zero, object is touching image edge and thus requires zero-padding
            # This ensures that Canny filter generates a closed surface
            if np.any(image_border):
                zslice = np.pad(zslice, 1)
                padded_zslice_count += 1
                padded = True

            # Perform edge detection with Canny filter (see skimage documentation)
            edges = canny(zslice)

            # Fill to generate solid object
            edges = binary_fill_holes(edges)

            # Remove padding
            if padded:
                edges = remove_border(edges)

            edges_canny[i, :, :] = edges

    # Convert to 8-bit image
    edges_canny = (edges_canny * 255).astype(np.uint8)

    # Filter out small objects that are smaller than radius of expansion and convert to labelmap
    edges_canny = label(remove_small_objects(edges_canny, iterations))

    return edges_canny, padded_zslice_count


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


def clean_binary_image(image_binary, sigma2d, small_objects_threshold):
    """
    Clean up an input binary image (0/1) by filling holes, removing small objects (below small_objects_threshold),
    and performing gaussian blur (with sigma2d).
    Performed in 2D on each zslice of image.
    Return cleaned up binary (0-1) image, float 64, same shape as input array.
    """
    cleaned = np.zeros_like(image_binary, dtype=np.float64)

    # Iterate over each zslice in image
    for i, zslice in enumerate(image_binary):
        zslice = binary_fill_holes(zslice)
        zslice = remove_small_objects(zslice, small_objects_threshold)
        zslice = (zslice * 255).astype(np.uint8)  # convert 0-255
        zslice = gaussian(zslice, sigma=sigma2d, preserve_range=False)  # output is 0-1
        cleaned[i, :, :] = zslice

    return cleaned


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


def run_label_fusion(seg, expandby_factor, sigma, pixmeta, canny_threshold):
    """
    Main function for running label fusion. Used in Surface Mesh Multiscale task to generate organoid label from
    single-cell segmentation.
    """
    seg_binary, expandby_pix, iterations = fuse_labels(seg, expandby_factor)
    blurred, anisotropic_sigma = anisotropic_gaussian_blur(seg_binary, sigma, pixmeta)
    edges_canny, padded_zslice_count = find_edges(blurred, canny_threshold, iterations)

    return edges_canny, expandby_pix, iterations, anisotropic_sigma, padded_zslice_count


def run_thresholding(
    raw_image,
    intensity_threshold,
    gaus_sigma_raw_img,
    gaus_sigma_thresh_img,
    small_objects_diameter,
    canny_threshold,
    pixmeta_raw,
    seg,
):
    """
    Main function for running intensity-based thresholding.
    Used in Segment by Intensity Threshold Fractal task to generate 3D label map from intensity image.
    """
    # Apply 3D gaussian blur to raw intensity image prior to thresholding
    blurred, anisotropic_sigma = anisotropic_gaussian_blur(
        raw_image, gaus_sigma_raw_img, pixmeta_raw, convert_to_8bit=False
    )
    # Generate binary mask
    # Simple intensity threshold; values above threshold set to 1, values below set to 0
    combo_binary = np.where(blurred > intensity_threshold, 1, 0)

    # Clean up binary mask by filling holes, removing small objects, and gaussian blur by z-slice
    small_objects_2dthreshold = np.pi * (small_objects_diameter / 2) ** 2
    cleaned = clean_binary_image(
        combo_binary, gaus_sigma_thresh_img, small_objects_2dthreshold
    )

    # Smoothen edges of mask via Canny edge detection
    small_objects_3dthreshold = (4 / 3) * np.pi * (small_objects_diameter / 2) ** 3
    contour, padded_zslice_count = find_edges(
        cleaned, canny_threshold, small_objects_3dthreshold
    )

    # Filter by organoid label image from 2D segmentation (converted to 3D)
    # This removes debris that is far from organoid (only in xy)
    # and at least partially removes touching neighboring organoids
    contour = contour * seg

    # Discard small disconnected components
    contour, roi_count = select_largest_component(contour)

    return contour, padded_zslice_count, roi_count
