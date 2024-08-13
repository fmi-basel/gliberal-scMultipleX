# Copyright (C) 2024 Friedrich Miescher Institute for Biomedical Research

##############################################################################
#                                                                            #
# Author: Nicole Repina              <nicole.repina@fmi.ch>                  #
#                                                                            #
##############################################################################

import numpy as np
from scipy.ndimage import binary_fill_holes, binary_erosion
from skimage.morphology import disk, remove_small_objects
from skimage.segmentation import expand_labels
from skimage.feature import canny
from skimage.filters import gaussian
from skimage.measure import label

from scmultiplex.features.FeatureFunctions import mesh_equivalent_diameter
from scmultiplex.meshing.FilterFunctions import calculate_mean_volume, load_border_values, remove_border


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
        seg_binary[i, :, :] = binary_erosion(zslice, disk(1), iterations=iterations)  # returns binary image, max val 1

    return seg_binary, expandby_pix, iterations


def anisotropic_gaussian_blur(seg_binary, sigma, pixmeta):
    """
    Perform gaussian blur of binary 3D image with anisotropic sigma. Sigma anisotropy calculated from pixel spacing.
    """
    # Convert binary segmentation into 8-bit image
    seg_fill_8bit = (seg_binary * 255).astype(np.uint8)

    # Scale sigma to match pixel anisotropy
    spacing = np.array(pixmeta)/pixmeta[1]  # (z, y, x) where x,y is normalized to 1 e.g. (3, 1, 1)
    anisotropic_sigma = tuple([sigma * (spacing[1] / x) for x in spacing])

    # Perform 3D gaussian blur
    # Output image has value range 0-1 (not always max 1)
    blurred = gaussian(seg_fill_8bit, sigma=anisotropic_sigma, preserve_range=False)

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


def run_label_fusion(seg, expandby_factor, sigma, pixmeta, canny_threshold):
    """
    Main function for running label fusion. Used in Surface Mesh Multiscale task to generate organoid label from
    single-cell segmentation.
    """
    seg_binary, expandby_pix, iterations = fuse_labels(seg, expandby_factor)
    blurred, anisotropic_sigma = anisotropic_gaussian_blur(seg_binary, sigma, pixmeta)
    edges_canny, padded_zslice_count = find_edges(blurred, canny_threshold, iterations)

    return edges_canny, expandby_pix, iterations, anisotropic_sigma, padded_zslice_count
