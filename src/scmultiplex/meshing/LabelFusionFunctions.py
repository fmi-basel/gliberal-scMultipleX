# Copyright (C) 2024 Friedrich Miescher Institute for Biomedical Research

##############################################################################
#                                                                            #
# Author: Nicole Repina              <nicole.repina@fmi.ch>                  #
#                                                                            #
##############################################################################
import logging

import dask
import dask.array as da
import numpy as np
import pandas as pd
from dask_image.ndmeasure import label as dask_label
from scipy.ndimage import (
    binary_erosion,
    binary_fill_holes,
    distance_transform_edt,
    generate_binary_structure,
)
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

logger = logging.getLogger(__name__)


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


def expand_3d_labels(label_image: np.ndarray, distance: int):
    """
    Expand labels in a 3D segmentation using Euclidean distance.

    Args:
        label_image (np.ndarray): 3D segmentation label image.
        distance (int): Distance (in pixels) to expand labels.

    Returns:
        np.ndarray: Expanded 3D label image.
    """
    # Get distance transform and the indices of nearest non-zero pixels
    distances, indices = distance_transform_edt(label_image == 0, return_indices=True)

    expanded = np.zeros_like(label_image)

    # Expand labels where distance is within threshold
    # Create a mask where distances are within the expansion threshold
    mask = distances <= distance
    # Use indexing to map the nearest label values
    expanded[mask] = label_image[tuple(idx[mask] for idx in indices)]
    return expanded


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


def fill_holes_by_slice(seg):
    """
    Fill holes in each 2D slice of a 3D binary array.

    This function processes a 3D array slice by slice along the first axis (Z-axis),
    filling any holes in each 2D slice. Holes are defined as background regions
    (zeros) that are completely surrounded by foreground (non-zero) pixels.

    Parameters
    ----------
    seg : array_like
        Input 3D binary array of shape (Z, Y, X). Must be exactly 3-dimensional. It can also be a 2D image with
        shape (1, Y, X).

        The array is expected to have non-zero values for foreground and zero for background.

    Returns
    -------
    seg_filled : np.ndarray
        A 3D array of the same shape as `seg` with holes filled in each slice.
    """
    if seg.ndim != 3:
        raise ValueError(f"Input array must be 3D, got shape {seg.shape}")

    # Initialize empty array
    seg_filled = np.zeros_like(seg)
    # Iterate over each zslice in image
    for i, zslice in enumerate(seg):
        # Fill holes
        seg_filled[i, :, :] = binary_fill_holes(zslice)
    return seg_filled


def fill_holes_by_slice_multi_instance(seg):
    """
    Fill holes in each labeled object of a 3D segmentation array, slice by slice.

    This function operates on a labeled 3D segmentation array where each non-zero
    integer represents a distinct object instance. For each label, a binary mask
    is created and holes are filled independently in each 2D slice (along the Z-axis).
    The filled masks are then recombined into a single labeled output array.

    This ensures that holes are filled **per instance**, preventing neighboring
    objects from being merged.

    Parameters
    ----------
    seg : array_like
        Input 3D label map of shape (Z, Y, X). Must be exactly 3-dimensional.
        It can also be a 2D image with shape (1, Y, X).
        Zero is treated as background, and all positive integer values are treated
        as separate object labels.

    Returns
    -------
    filled_array : np.ndarray
        A 3D labeled array of the same shape as `seg`, where holes have been
        filled independently for each object label on a per-slice basis.
    """
    unique_labels = list(set(seg.ravel()) - {0})  # drop 0 background
    filled_array = np.zeros_like(seg)
    for lab in unique_labels:
        binary_mask = seg == lab
        filled_mask = fill_holes_by_slice(binary_mask)
        # Assign filled mask to result with label
        filled_array[filled_mask] = lab
    return filled_array


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


def find_edges(
    cleaned, contour_value_outer, contour_value_inner, min_size, segment_lumen=False
):
    """
    Find edges of input 3D image by z-slice
    """
    # Initialize empty array
    outer_stack = np.zeros_like(cleaned)

    if segment_lumen:
        lumen_stack = np.zeros_like(cleaned)

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

        outer_contours = find_contours(
            zslice, level=contour_value_outer, fully_connected="high"
        )
        # if any contours are detected...
        if outer_contours:
            # Sort contours by length (assuming the two longest ones are the ones we need)
            outer_contours = sorted(outer_contours, key=len, reverse=True)

            # assume largest contour is contour of outer epithelium
            if len(outer_contours) > 0:
                outer_contour = outer_contours[0]
                # identify pixels belonging to inside of contour
                rr, cc = polygon(
                    outer_contour[:, 0], outer_contour[:, 1], shape=zslice.shape
                )
                outer[rr, cc] = 1  # Set pixels inside the polygon to 1

        if segment_lumen:
            inner_contours = find_contours(
                zslice, level=contour_value_inner, fully_connected="high"
            )
            # assume second-largest contour is lumen
            if len(inner_contours) > 1:
                lumen = np.zeros_like(zslice)
                inner_contour = inner_contours[1]
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
        # TODO remove hard-coded size factor here; allowing lumen debris to be 20x smaller than small object size
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

    # Iterate over each zslice in image
    for i, zslice in enumerate(image_binary):

        # Remove small objects prior to expansion
        zslice = zslice > 0
        zslice = remove_small_objects(zslice, small_objects_threshold)
        zslice = zslice.astype(int)

        if fill_holes:
            zslice = binary_fill_holes(zslice)
        zslice = expand_labels(
            zslice, expandby_pix
        )  # expand mask to fuse labels and hill holes
        if fill_holes:
            zslice = binary_fill_holes(zslice)
        zslice = binary_erosion(
            zslice,
            disk(expandby_pix),
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


def select_largest_component(label_image, connectivity=3):
    """
    Select largest connected component from a binary or label image.
    Discard all smaller disconnected components.
    Return binary 0/1 image.
    """

    # Convert to binary mask first
    binary = label_image > 0

    # Connected-component labeling
    cc = label(binary, connectivity=connectivity)

    rois = regionprops(cc)
    roi_count = len(rois)

    if roi_count == 0:
        return binary.astype(np.uint8), roi_count

    # Find largest connected component
    largest_region = max(rois, key=lambda r: r.area)
    largest_label = largest_region.label

    # Keep only largest component
    largest_component = cc == largest_label

    return largest_component.astype(np.uint8), roi_count


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
        blurred, canny_threshold, canny_threshold, expandby_pix, segment_lumen=False
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


def run_expansion(
    seg, expandby, expansion_distance_image_based=False, expand_in_z=False
):
    """
    Main function for running label expansion, used in Expand Labels task.
    """
    if expansion_distance_image_based:
        expandby_pix = get_expandby_pixels(seg, expandby_factor=expandby)
    else:
        expandby_pix = expandby

    if expand_in_z:
        seg_expanded = expand_3d_labels(seg, expandby_pix)
    else:
        seg_expanded = expand_2d_labels(seg, expandby_pix)
    return seg_expanded, expandby_pix


def run_thresholding(
    raw_image,
    threshold_type,
    gaus_sigma_raw_img,
    gaus_sigma_thresh_img,
    small_objects_diameter,
    expand_by_pixel,
    contour_value_outer,
    contour_value_inner,
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
            contour_value_outer,
            contour_value_inner,
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
            contour_value_outer,
            contour_value_inner,
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
    input_chunking = label_dask.chunks
    input_shape = label_dask.shape
    dtype = label_dask.dtype
    rank = label_dask.ndim

    # Set all values > 0 to True
    binary_dask = label_dask > 0  # binary boolean array

    if connectivity is None:
        structure = None
    else:

        if connectivity > binary_dask.ndim:
            raise ValueError(
                "Connectivity must not be greater than number of array dimensions. Check user input."
            )
        structure = generate_binary_structure(rank=rank, connectivity=connectivity)

        # Structuring element always has shape: (3, 3, 3)
        # If input image is a 2D image (1,y,x), the structuring element is adjusted...
        # ...so that only its middle z-slice has True values; above and below is set to False
        if input_shape[0] == 1:
            # Keep center Z slice, zero out everything else along first axis
            center = structure.shape[0] // 2  # center index along Z
            tmp = np.zeros_like(structure, dtype=bool)
            tmp[center, :, :] = structure[center, :, :]  # copy center Z slice
            structure = tmp

    # Run Dask-based connected components
    fused_dask, label_count = dask_label(binary_dask, structure=structure)

    # Cast to original dtype lazily
    fused_dask = fused_dask.astype(dtype)

    # Rechunk if necessary
    if fused_dask.chunks != input_chunking:
        logger.info("Rechunk: Rechunking fused dask array to match input chunks...")
        logger.info(
            f"Rechunk: Fused dask array shape before rechunk: {fused_dask.shape}"
        )
        logger.info(f"Rechunk: Fused dask chunks before rechunk: {fused_dask.chunks}")
        logger.info(f"Rechunk: Input shape: {input_shape}")
        logger.info(f"Rechunk: Input chunks: {input_chunking}")
        fused_dask = fused_dask.rechunk(input_chunking)
        logger.info(
            f"Rechunk: Fused dask array chunks after rechunk: {fused_dask.chunks}"
        )

    return fused_dask, label_count, rank


def get_label_mapping_from_block(
    block1: np.ndarray,
    block2: np.ndarray,
) -> dict:
    """
    Build a mapping from nonzero values in a single block of array_1 to
    the corresponding values in array_2, recording only the first occurrence
    of each label within the block.

    This function operates on NumPy array chunks and is intended to be
    used with Dask's delayed computation to process large arrays without
    loading the entire array into memory.

    Parameters
    ----------
    block1 : np.ndarray
        A chunk of the first array (e.g., a label array). Can be 2D or 3D.
        Nonzero values are considered foreground labels; zeros are ignored.

    block2 : np.ndarray
        A chunk of the second array, of the same shape as `block1`.
        Each nonzero pixel in `block1` is mapped to the corresponding value
        in `block2`.

    Returns
    -------
    mapping : dict
        Dictionary mapping each unique nonzero value in `block1` to its
        corresponding value in `block2` based on the first occurrence in the block.
        Keys and values are Python integers.

    Notes
    -----
    - Only the first occurrence of each key in the block is recorded.
    - Works efficiently with small or large array chunks.
    - Does not operate on full Dask arrays directly.

    Example
    -------
    >> block1 = np.array([[0, 1], [2, 1]])
    >> block2 = np.array([[0, 10], [20, 15]])
    >> get_label_mapping_from_block(block1, block2)
    {1: 10, 2: 20}
    """
    mapping = {}
    pix_nonzero = np.nonzero(block1)  # tuple of arrays (z, y, x)

    # Iterate over non-zero pixels
    for idx in zip(*pix_nonzero):
        key = block1[idx].item()
        value = block2[idx].item()
        if key not in mapping:
            mapping[key] = value

    return mapping


def get_label_mapping_dask(
    array_1: da.array,
    array_2: da.array,
) -> dict:
    """
    Build a global mapping from values in `array_1` to corresponding values
    in `array_2` by processing large arrays in chunks with Dask.

    Each unique nonzero value in `array_1` is mapped to its corresponding
    value in `array_2` based on the first occurrence. Uses Dask delayed
    computation to merge per-chunk mappings into a single global dictionary.

    Parameters
    ----------
    array_1 : dask.array.Array
        The first array (e.g., a label array). Nonzero values are considered
        foreground labels; zeros are ignored.
        Must be the same shape as `array_2`.

    array_2 : dask.array.Array
        The second array, of the same shape as `array_1`. Provides the values
        corresponding to each label in `array_1`.

    Returns
    -------
    global_mapping : dict
        Dictionary mapping each unique nonzero value in `array_1` to the
        corresponding value in `array_2` based on the first occurrence across
        all chunks. Keys and values are Python integers.

    Notes
    -----
    - This function is memory-efficient because it processes each chunk independently.
    - Each chunk returns a dictionary of mappings, and the final global
      mapping preserves the first occurrence across the entire array.
    - Works for 2D or 3D arrays.
    - Dask chunking must align between `array_1` and `array_2` so that corresponding
      pixels are in the same chunk.
    - Returns a fully computed Python dict; large arrays themselves are not loaded into memory.

    Example
    -------
    >> a1 = da.from_array(np.array([[0, 1], [2, 1]]), chunks=(1, 2))
    >> a2 = da.from_array(np.array([[0, 10], [20, 15]]), chunks=(1, 2))
    >> get_label_mapping_dask(a1, a2)
    {1: 10, 2: 20}
    """
    if array_1.chunks != array_2.chunks:
        raise ValueError(
            f"Chunks of array_1 {array_1.chunks} and array_2 {array_2.chunks} do not match. "
            "Dask arrays must have the same chunking for label mapping. Were unfused and fused images written by "
            "different zarr pyramid writers?"
        )
    # Convert Dask array chunks to delayed objects
    blocks_1 = array_1.to_delayed().ravel()
    blocks_2 = array_2.to_delayed().ravel()

    # Apply per-block mapping function
    delayed_dicts = [
        dask.delayed(get_label_mapping_from_block)(b1, b2)
        for b1, b2 in zip(blocks_1, blocks_2)
    ]

    # Compute all chunk-level dicts
    block_mappings = dask.compute(*delayed_dicts)

    # Merge all block dicts into a single global mapping
    global_mapping = {}
    for d in block_mappings:
        for k, v in d.items():
            if k not in global_mapping:
                global_mapping[k] = v

    return global_mapping
