# Copyright (C) 2023 Friedrich Miescher Institute for Biomedical Research

##############################################################################
#                                                                            #
# Author: Nicole Repina              <nicole.repina@fmi.ch>                  #
# Author: Manan Lalit                <lalit@mpi-cbg.de>                      #
# Author: Enrico Tagliavini          <enrico.tagliavini@fmi.ch>              #
#                                                                            #
##############################################################################


import os.path
import sys
from copy import deepcopy
from typing import Set, Tuple

import dask.array as da
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

# this is an hack to run platymatch without modifying its code. In some parts of the code
# platymatch will import other submodules from itself in an absolute way (i.e. from platymatch.xxx import ....)
# rather than a relative way. Maybe we can create a pull request to switch those import statements to relative
import scmultiplex

sys.path.append(os.path.join(scmultiplex.__path__[0], r"platymatch"))

from platymatch.estimate_transform.apply_transform import (  # noqa: E402
    apply_affine_transform,
)
from platymatch.estimate_transform.perform_icp import perform_icp  # noqa: E402
from platymatch.estimate_transform.shape_context import (  # noqa: E402
    do_ransac_complete,
    get_unary,
)
from platymatch.utils.utils import (  # noqa: E402
    compute_average_bg_intensity,
    generate_ffd_transformed_image,
    get_centroid,
    get_mean_distance,
    normalize_min_max_percentile,
)

# warnings.filterwarnings("ignore")


def calculate_stats(pc, column=-1):
    """
    Calculate mean and standard deviation of nuclear volumes in organoid
    Helper function used during affine linking
    """
    mean = np.mean(pc[:, column].transpose())
    std = np.std(pc[:, column].transpose())
    return mean, std


def calculate_size(mean, std):
    """
    Calculate lower size cutoff of distribution
    Helper function used during affine linking
    """
    if (mean - 3 * std) > 0:
        size = mean - 3 * std
    else:
        size = mean - 2 * std
    return size


def calculate_quantiles(pc, q=0.5, column=-1):
    """
    Calculate mean and standard deviation of nuclear volumes in organoid
    Helper function used during affine linking
    q is quantile in range [0,1] to calclulate
    """
    quantile = np.quantile(pc[:, column].transpose(), q)
    iqr = np.quantile(pc[:, column].transpose(), 0.75) - np.quantile(
        pc[:, column].transpose(), 0.25
    )
    return quantile, iqr


def calculate_nucleus_size_non_normal(moving_pc, fixed_pc, q=0.3):
    """
    Calculate lower size cutoff of nuclear volumes in organoid in fixed and moving image
    Helper function used during affine linking
    Uses medians and IQR to improve stability for not normal data inputs
    """
    moving_nuclei_size, _ = calculate_quantiles(moving_pc, q, column=-1)
    fixed_nuclei_size, _ = calculate_quantiles(fixed_pc, q, column=-1)

    return moving_nuclei_size, fixed_nuclei_size


def calculate_nucleus_size(moving_pc, fixed_pc):
    """
    Calculate lower size cutoff of nuclear volumes in organoid in fixed and moving image
    Helper function used during affine linking
    """
    moving_nuclei_size_mean, moving_nuclei_size_std = calculate_stats(
        moving_pc, column=-1
    )
    moving_nuclei_size = calculate_size(moving_nuclei_size_mean, moving_nuclei_size_std)

    fixed_nuclei_size_mean, fixed_nuclei_size_std = calculate_stats(fixed_pc, column=-1)
    fixed_nuclei_size = calculate_size(fixed_nuclei_size_mean, fixed_nuclei_size_std)

    return moving_nuclei_size, fixed_nuclei_size


def run_affine(moving_pc, fixed_pc, ransac_iterations, icp_iterations):
    """
    Run affine linking of two point clouds
    :moving_pc: numpy array of label centroids and volume of moving point cloud, e.g. RX
        rows are unique label ID and columns are in order:
            "label_id", "x_centroid", "y_centroid", "z_centroid", "volume"
        all units are in pixels
        scaling must match label image, i.e. z_centroid and volume do not take into account pixel anisotropy
    :fixed_pc: numpy array of label centroids and volume of fixed point cloud, e.g. R0
        same specifications as for moving_pc
    :ransac_iterations: integer number of RANSAC iterations
    :icp_iterations: integer number of ICP iterations
    :return:
        results_affine: numpy array of results, with column order:
            "fixed_label_id", "moving_label_id", "euclidian_distance_pix_fixed_to_affine", "confidence"
            euclidian_distance_pix_fixed_to_affine measures the distance between centroids for each matched pair of
            nuclei in affine-transformed moving and untransformed fixed labels.
        transform_affine: affine transformation matrix, for use in FFD linking or for applying image transformation
    """
    # Obtain ids
    moving_ids = moving_pc[:, 0].transpose()  # (N,)
    moving_detections = np.flip(moving_pc[:, 1:-1], 1).transpose()  # z y x, 3 x N

    fixed_ids = fixed_pc[:, 0].transpose()  # (N,)
    fixed_detections = np.flip(fixed_pc[:, 1:-1], 1).transpose()  # z y x, 3 x N

    # Calculate nucleus size
    moving_nuclei_size, fixed_nuclei_size = calculate_nucleus_size_non_normal(
        moving_pc, fixed_pc, q=0.3
    )
    # ransac error should be roughly cell diameter, in pixels
    ransac_error = 0.5 * (moving_nuclei_size ** (1 / 3) + fixed_nuclei_size ** (1 / 3))

    # Determine centroids
    moving_centroid = get_centroid(moving_detections, transposed=False)
    fixed_centroid = get_centroid(fixed_detections, transposed=False)

    # Get Average Mean Distance
    moving_mean_distance = get_mean_distance(moving_detections, transposed=False)
    fixed_mean_distance = get_mean_distance(fixed_detections, transposed=False)

    # Generate Unaries
    unary_11, unary_12, _, _ = get_unary(
        moving_centroid,
        mean_distance=moving_mean_distance,
        detections=moving_detections,
        type="moving",
        transposed=False,
    )
    unary_21, unary_22, unary_23, unary_24 = get_unary(
        fixed_centroid,
        mean_distance=fixed_mean_distance,
        detections=fixed_detections,
        type="fixed",
        transposed=False,
    )

    # Generate distance matrices
    U11 = -np.matmul(unary_11, unary_21.transpose())
    U12 = -np.matmul(unary_11, unary_22.transpose())
    U13 = -np.matmul(unary_11, unary_23.transpose())
    U14 = -np.matmul(unary_11, unary_24.transpose())
    U21 = -np.matmul(unary_12, unary_21.transpose())
    U22 = -np.matmul(unary_12, unary_22.transpose())
    U23 = -np.matmul(unary_12, unary_23.transpose())
    U24 = -np.matmul(unary_12, unary_24.transpose())

    # Perform RANSAC
    transform_matrix_shape_context, inliers = do_ransac_complete(
        U11,
        U12,
        U13,
        U14,
        U21,
        U22,
        U23,
        U24,
        moving_detections,
        fixed_detections,
        ransac_samples=4,
        ransac_iterations=ransac_iterations,
        ransac_error=ransac_error,
    )

    sign_id = np.argmax(inliers)

    transformed_moving_detections = apply_affine_transform(
        moving_detections, transform_matrix_shape_context
    )
    transform_matrix_icp, icp_residuals = perform_icp(
        transformed_moving_detections, fixed_detections, icp_iterations, "Affine"
    )
    transform_affine = np.matmul(transform_matrix_icp, transform_matrix_shape_context)

    # Apply Affine Transform
    transformed_moving_detections = apply_affine_transform(
        moving_detections, transform_affine
    )  # 3 x N
    cost_matrix = cdist(
        transformed_moving_detections.transpose(), fixed_detections.transpose()
    )

    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # Save to csv file, the predicted matches ids and confidence
    if sign_id == 0:
        confidence = -U11
    elif sign_id == 1:
        confidence = -U12
    elif sign_id == 2:
        confidence = -U13
    elif sign_id == 3:
        confidence = -U14
    elif sign_id == 4:
        confidence = -U21
    elif sign_id == 5:
        confidence = -U22
    elif sign_id == 6:
        confidence = -U23
    elif sign_id == 7:
        confidence = -U24

    results_affine = np.column_stack(
        (
            fixed_ids[col_indices].transpose(),
            moving_ids[row_indices].transpose(),
            cost_matrix[row_indices, col_indices],
            confidence[row_indices, col_indices],
        )
    )

    return results_affine, transform_affine


def run_ffd(
    moving_pc,
    fixed_pc,
    moving_transformed_affine_raw_image,
    moving_transformed_affine_label_image,
    fixed_raw_image,
    fixed_label_image,
):
    """
    Run free form deformation of affine-transformed images to calculate FFD transformation matrix and linking
    Moving images should be pre-aligned with fixed image via affine transform e.g. with generate_affine_transformed_image
    :moving_pc: numpy array of label ids in moving point cloud, e.g. RX
        rows are unique label id and first column must correspond to the label_id
        note in this function only label_id column is used and centroids and volume are irrelevant
    :fixed_pc: numpy array of label centroids and volume of fixed point cloud, e.g. R0
        rows are unique label id and columns are in order:
            "label_id", "x_centroid", "y_centroid", "z_centroid", "volume"
        all units are in pixels
        scaling must match label image, i.e. z_centroid and volume do not take into account pixel anisotropy
    :moving_transformed_affine_raw_image: numpy array of moving raw intensity image that has been affine-transformed, e.g. RX raw affine
    :moving_transformed_affine_label_image: numpy array of moving label map image that has been affine-transformed, e.g. RX seg affine
    :fixed_raw_image: numpy array of fixed raw image, e.g. R0 raw
    :fixed_label_image: numpy array of fixed label map image, e.g. R0 seg
    :return:
        results_ffd: numpy array of results, with column order:
            "fixed_label_id", "moving_label_id", "euclidian_distance_pix_fixed_to_ffd"
            euclidian_distance_pix_fixed_to_ffd measures the distance between centroids for each matched pair of
            nuclei in ffd-transformed moving and untransformed fixed labels.
        transform_ffd: ffd transformation matrix, for use in applying image transformation to raw image
        transformed_ffd_label_image: numpy array, result of ffd transformation of input moving affine image
    """

    fixed_ids = fixed_pc[:, 0].transpose()
    fixed_detections = np.flip(fixed_pc[:, 1:-1], 1)

    # Normalize fixed and moving-affine image
    fixed_raw_image_normalized = normalize_min_max_percentile(
        fixed_raw_image, 1, 99.8, axis=(0, 1, 2)
    )
    moving_transformed_affine_raw_image_normalized = normalize_min_max_percentile(
        moving_transformed_affine_raw_image, 1, 99.8, axis=(0, 1, 2)
    )

    # Generate Free Form Deformation transform (based on Intensity Correlation) --> Note this may take some time
    transform_ffd = generate_ffd_transformed_image(
        fixed_image=sitk.GetImageFromArray(
            fixed_raw_image_normalized.astype(np.float32)
        ),
        moving_image=sitk.GetImageFromArray(
            moving_transformed_affine_raw_image_normalized.astype(np.float32)
        ),
    )

    # Generate FFD-transformed label image
    transformed_ffd_label_image_sitk = sitk.Resample(
        sitk.GetImageFromArray(moving_transformed_affine_label_image),
        sitk.GetImageFromArray(fixed_raw_image),
        transform_ffd,
        sitk.sitkNearestNeighbor,
        0.0,
        sitk.GetImageFromArray(moving_transformed_affine_label_image).GetPixelID(),
    )
    transformed_ffd_label_image = sitk.GetArrayFromImage(
        transformed_ffd_label_image_sitk
    ).astype(fixed_label_image.dtype)

    # obtain accuracy after performing FFD
    ids = moving_pc[:, 0].transpose()  # same as moving_ids in run_affine functions
    moving_ffd_ids = []
    transformed_moving_ffd_detections = []

    for id in ids:  # for id in original moving label image...
        # find centroid for same id in FFD-transformed label image
        z, y, x = np.where(transformed_ffd_label_image == id)
        zm, ym, xm = np.mean(z), np.mean(y), np.mean(x)
        if len(z) == 0:
            zm = ym = xm = 0.0
        transformed_moving_ffd_detections.append(np.array([zm, ym, xm]))  # N x 3
        moving_ffd_ids.append(id)
    # calculate euclidian distance btw centroids of FFD-transformed label image and centroids of original fixed image
    cost_matrix = cdist(transformed_moving_ffd_detections, fixed_detections)

    moving_ffd_ids = np.array(moving_ffd_ids)
    # assign pairs of nuclei by minimizing total cost of distance matrix with LSAP
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # Return matching ids
    results_ffd = np.column_stack(
        (
            fixed_ids[col_indices].transpose(),
            moving_ffd_ids[row_indices].transpose(),
            cost_matrix[row_indices, col_indices],
        )
    )

    return results_ffd, transform_ffd, transformed_ffd_label_image


def generate_ffd_rawimage_from_affine(
    moving_transformed_affine_raw_image,
    fixed_raw_image,
    fixed_label_image,
    transform_ffd,
):
    """
    Apply ffd transform to intensity image. Intended to be applied to moving image that has been affine-transformed.
    :moving_transformed_affine_raw_image: numpy array of moving raw intensity image that has been affine-transformed, e.g. RX raw affine
    :fixed_raw_image: numpy array of fixed raw image, e.g. R0 raw
    :fixed_label_image: numpy array of fixed label map image, e.g. R0 seg
    :transform_ffd: ffd transformation matrix e.g. from output of run_ffd
    :return:
        moving_transformed_ffd_raw_image: numpy array, result of ffd transformation of input moving affine image
        Note: image has some intensity normalization internally - ask Manan, TO-DO
    """
    transformed_ffd_raw_image_sitk = sitk.Resample(
        sitk.GetImageFromArray(moving_transformed_affine_raw_image),
        sitk.GetImageFromArray(fixed_raw_image),
        transform_ffd,
        sitk.sitkLinear,
        compute_average_bg_intensity(fixed_raw_image, fixed_label_image),
        sitk.GetImageFromArray(moving_transformed_affine_raw_image).GetPixelID(),
    )

    moving_transformed_ffd_raw_image = sitk.GetArrayFromImage(
        transformed_ffd_raw_image_sitk
    ).astype(fixed_raw_image.dtype)

    return moving_transformed_ffd_raw_image


def relabel_RX_numpy(
    rx_seg,
    matches,
    moving_colname="RX_nuc_id",
    fixed_colname="R0_nuc_id",
    daskarr=False,
):
    """
    Relabel RX label map to match R0 labels based on linking. Matches is affine or ffd pandas df after platymatch matching
    """
    # key is moving_label, value is fixed_label
    matching_dict = make_linking_dict(matches, moving_colname, fixed_colname)
    # convert to numpy if input is not numpy (e.g. if input is a dask array)
    if not isinstance(rx_seg, np.ndarray):
        rx_seg = np.asarray(rx_seg)

    rx_numpy_matched = np.zeros_like(rx_seg)

    # identify indexes that are non-zero
    rx_seg_nonzero = np.nonzero(rx_seg)

    # for each nonzero index, relabel if key is in dictionary
    labels_in_input = set()
    labels_in_output = set()
    for nonzero_pixel in zip(*rx_seg_nonzero):

        key = rx_seg[nonzero_pixel].item()  # fetch value of given pixel
        labels_in_input.add(key)  # add key to set

        try:
            relabeled_value = matching_dict[key]
        except KeyError:
            pass
        else:
            labels_in_output.add(relabeled_value)
            rx_numpy_matched[tuple(nonzero_pixel)] = relabeled_value

    count_input = len(labels_in_input)
    count_output = len(labels_in_output)

    if daskarr:
        rx_numpy_matched = da.from_array(rx_numpy_matched)

    return rx_numpy_matched, count_input, count_output, labels_in_output


# new functions for relinking
def count_number_of_labels_in_dask(label_dask: da.Array) -> Tuple[int, Set[int]]:
    """
    Return (1) number of unique labels in dask array and (2) the set of those labels.
    All labels except 0 are included. Input dask array must have only non-negative positive integers.
    """

    bcounts = da.bincount(label_dask.ravel())

    existing_labels = (
        (bcounts > 0).compute().nonzero()[0]
    )  # compute array and count 'True' labels with .nonzero()
    existing_labels = existing_labels[existing_labels != 0]  # drop 0 label

    existing_labels_set = set(existing_labels)
    number_of_labels = len(existing_labels_set)

    return number_of_labels, existing_labels_set


def make_correction_dict(
    matches: pd.DataFrame,
    moving_colname: str = "RX_nuc_id",
    fixed_colname: str = "R0_nuc_id",
) -> dict:

    # key is moving_label (current label), value is fixed_label (value to rename to)
    matching_dict = make_linking_dict(matches, moving_colname, fixed_colname)

    return matching_dict


def make_relabeled_block(
    block1: np.ndarray,
    matching_dict: dict,
    label_dtype: np.dtype,
) -> np.ndarray:

    # return indeces of elements that are non-zero in block as tuple
    pix_nonzero = np.nonzero(block1)

    # initialize new block
    relabeled_block = np.zeros_like(block1, dtype=label_dtype)

    for nonzero_pixel in zip(*pix_nonzero):
        # nonzero_pixel is [z,y,x]
        # fetch value (as Python scalar) of array at given pixel, as string to match obs

        key = block1[nonzero_pixel].item()  # fetch value of given pixel

        try:
            relabeled_value = matching_dict[key]
        except KeyError:
            pass
        else:
            relabeled_block[tuple(nonzero_pixel)] = relabeled_value

    return relabeled_block


def run_relabel_dask(
    label_dask: da.Array,
    matches: pd.DataFrame,
    moving_colname: str = "RX_nuc_id",
    fixed_colname: str = "R0_nuc_id",
) -> da.Array:

    matching_dict = make_correction_dict(matches, moving_colname, fixed_colname)

    dtype = label_dask.dtype

    relabeled_dask = da.map_blocks(
        make_relabeled_block,
        label_dask,
        matching_dict=matching_dict,
        label_dtype=dtype,
        dtype=dtype,
    )

    return relabeled_dask


def remove_labels(seg_img, labels_to_remove, datatype):
    """
    Remove labels from segmentation image.
    labels_to_remove is list of labels, and each value should match data format of
    """
    seg_img_relabeled = deepcopy(seg_img)

    for lab in labels_to_remove:
        lab = lab.astype(datatype)
        seg_img_relabeled[seg_img_relabeled == lab] = 0

    return seg_img_relabeled


def make_linking_dict(matches, moving_colname, fixed_colname):
    linking_dict = matches.set_index(moving_colname).T.to_dict("index")[fixed_colname]
    return linking_dict
