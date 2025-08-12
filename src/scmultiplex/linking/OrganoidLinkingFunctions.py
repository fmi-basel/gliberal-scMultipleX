# Copyright (C) 2023 Friedrich Miescher Institute for Biomedical Research

##############################################################################
#                                                                            #
# Author: Nicole Repina              <nicole.repina@fmi.ch>                  #
# Author: Enrico Tagliavini          <enrico.tagliavini@fmi.ch>              #
#                                                                            #
##############################################################################

from typing import Tuple

import dask.array as da
import numpy as np
import pandas as pd
from scipy.ndimage import affine_transform, shift
from skimage.measure import regionprops_table
from skimage.registration import phase_cross_correlation
from skimage.transform import EuclideanTransform

from scmultiplex.linking.matching import matching


def pad_img_set(img1, img2):
    """
    Zero-pad two images to be the same size
    :param img1: 2D numpy array, typically R0
    :param img2: 2D numpy array, typically RX
    :return: padded img1 and img2
    """
    img1_padded = np.pad(
        img1,
        [
            (0, max(0, img2.shape[0] - img1.shape[0])),
            (0, max(0, img2.shape[1] - img1.shape[1])),
        ],
        mode="constant",
        constant_values=0,
    )

    img2_padded = np.pad(
        img2,
        [
            (0, max(0, img1.shape[0] - img2.shape[0])),
            (0, max(0, img1.shape[1] - img2.shape[1])),
        ],
        mode="constant",
        constant_values=0,
    )

    return img1_padded, img2_padded


def binarize_img(img):
    """
    Binarize image, any values above 0 are set to 1
    :param img: 2D numpy array
    :return: binarized 2D numpy array (values 0 or 1)
    """
    img = img.copy()  # copy to not change original variable
    img[img > 0] = 1
    return img


def subsample_image(img, bin):
    """
    Subsample image by indicated bin spacing
    :param img: 2D numpy array of labeled objects
    :bin: integer value for subsampling, ex. bin = 4 means every 4th pixel is sampled
    :return: subsampled 2D numpy array with size reduced by bin^2
    """
    img_bin = img[::bin, ::bin]
    return img_bin


def calculate_shift(img0, imgX, bin):
    """
    Calculate xy shift between a set of 2D images based on phase cross correlation for image registration
    :param img0: 2D numpy array of labeled objects, reference image ex. R0
    :param imgX: 2D numpy array of labeled objects, moving image ex. RX
    :bin: integer value for subsampling, ex. bin = 4 means every 4th pixel is sampled
    :return:
        shifts: ndarray shift vector (y,x) in pixels required to shift moving image (imgX) relative to reference image (img0)
            pixel shift is relative to input imgX scaling (i.e. before subsampling)
        img0_pad: img0 padded to be same dimensions as imgX_pad
        imgX_pad: imgX padded to be same dimensions as img0_pad
    """

    # pad so that images have same shape
    img0_pad, imgX_pad = pad_img_set(img0, imgX)

    if (img0_pad.shape[0] != imgX_pad.shape[0]) | (
        img0_pad.shape[1] != imgX_pad.shape[1]
    ):
        raise ValueError("image pair must have same dimensions")

    # binarize padded overviews
    img0_pad_binary, imgX_pad_binary = binarize_img(img0_pad), binarize_img(imgX_pad)

    # subsample so that registration runs faster
    img0_pad_binary_bin, imgX_pad_binary_bin = subsample_image(
        img0_pad_binary, bin=bin
    ), subsample_image(imgX_pad_binary, bin=bin)

    # calculate shifts and take into account subsampling
    result = phase_cross_correlation(img0_pad_binary_bin, imgX_pad_binary_bin)
    shifts = bin * result[0]  # (y,x)

    return shifts, img0_pad, imgX_pad


def apply_shift(img, shifts):
    """
    Apply xy shift to image to perform registration, typically applied to moving image RX
    :param img: 2D numpy array
    :shifts: ndarray shift vector (y,x) in pixel units
    :return: shifted 2D numpy array, with empty regions from shift filled with 0
    """

    img_shifted = shift(img, shifts, mode="constant", cval=0)

    return img_shifted


def calculate_matching(img0, imgX, iou_cutoff):
    """
    Calculate matching between two label images based on intersection over union score
    :param img0: 2D numpy array of labeled objects, reference image ex. R0
    :param imgX: 2D numpy array of labeled objects, moving image ex. RX
    :iou_cutoff: float between 0 and 1 to specify intersection over union cutoff.
        Linked organoid pairs that have an iou below this value are filtered out
    :return:
        stat: Matching object (tuple of named tuples) that includes various accuracy and confidence metrics
        df: Pandas DataFrame of all detected organoid matches
        df_filt: Pandas DataFrame of organoid matches filtered by iou_cutoff, typically used for downstream linking.
    """
    stat = matching(img0, imgX, criterion="iou", thresh=iou_cutoff, report_matches=True)

    df = pd.DataFrame(
        list(zip([x[0] for x in stat[14]], [x[1] for x in stat[14]], stat[15])),
        columns=["R0_label", "RX_label", "iou"],
    )
    df_filt = df[df["iou"] > iou_cutoff]

    return stat, df, df_filt


def shift_array_3d_dask(arr, shift):
    """
    Shift a 3D Dask array along (z, y, x) axes.
    Pads with zeros and crops to original shape.

    # To trigger computation, use like this:
    # shifted = shift_array_3d_dask(arr, (0, 10, -20))
    # result = shifted.compute() # trigger compute

    Parameters:
        arr (dask.array): Input 3D Dask array.
        shift (tuple): (dz, dy, dx) shift values.

    Returns:
        dask.array: Shifted array with same shape.
    """
    assert arr.ndim == 3, "Only 3D arrays supported"
    assert len(shift) == 3

    orig_shape = arr.shape

    # Compute pad at beginning or end of array
    pad_width = []

    # Compute cropping slices to get back to original shape
    slices = []
    for axis, s in enumerate(shift):
        if s > 0:
            pad = (s, 0)  # pad at start
            slices.append(slice(0, orig_shape[axis]))
        elif s < 0:
            pad = (0, -s)  # pad at end
            slices.append(slice(-s, -s + orig_shape[axis]))
        else:
            pad = (0, 0)
            slices.append(slice(0, orig_shape[axis]))
        pad_width.append(pad)

    # Pad with zeros (no compute)
    padded = da.pad(arr, pad_width=pad_width, mode="constant", constant_values=0)

    # Crop to retain same shape as input array (no computed)
    shifted = padded[tuple(slices)]

    return shifted


def resize_array_to_shape(arr, target_shape):
    """
    Resize a 2D or 3D array to match a given target shape.
    Crops or pads only on the high (max index) end of each axis.

    Parameters
    ----------
    arr : np.ndarray or dask.array
        Input array. Must be either 2D (Y, X) or 3D (Z, Y, X).

    target_shape : tuple of int
        Desired output shape. Must match number of dimensions of `arr`.

    Returns
    -------
    resized : np.ndarray or dask.array
        Resized array with shape matching `target_shape`.

    Raises
    ------
    ValueError
        If input array and target shape do not have the same number of dimensions,
        or if resizing fails to produce the expected shape.
    """
    if arr.ndim != len(target_shape):
        raise ValueError(
            "Input array and target_shape must have the same number of dimensions"
        )

    curr_shape = arr.shape
    slices = []
    pad_width = []

    # Determine slices and padding for each dimension
    for i in range(arr.ndim):
        diff = target_shape[i] - curr_shape[i]
        if diff < 0:
            # Crop from end
            slices.append(slice(0, target_shape[i]))
            pad_width.append((0, 0))
        else:
            # Pad at end
            slices.append(slice(0, curr_shape[i]))
            pad_width.append((0, diff))

    # Crop first
    arr_cropped = arr[tuple(slices)]

    # Pad if needed
    if any(p > 0 for _, p in pad_width):
        if isinstance(arr, da.Array):
            arr_resized = da.pad(
                arr_cropped, pad_width=pad_width, mode="constant", constant_values=0
            )
        else:
            arr_resized = np.pad(
                arr_cropped, pad_width=pad_width, mode="constant", constant_values=0
            )
    else:
        arr_resized = arr_cropped

    # Final shape check
    if arr_resized.shape != target_shape:
        raise ValueError("Final array does not match target shape.")

    return arr_resized


def get_sorted_label_centroids(img: np.ndarray) -> np.ndarray:
    """
    Compute and return centroids of labeled regions in a 2D label image,
    sorted by increasing label value.

    Parameters
    ----------
    img : np.ndarray
        A 2D NumPy array containing integer label values. Each unique
        non-zero value is treated as a separate labeled region.

    Returns
    -------
    sorted_centroids : np.ndarray of shape (N, 2)
        A NumPy array of (x, y) coordinates (i.e., (col, row)) representing
        the centroids of the labeled regions, sorted in ascending order of their labels.
        Each row corresponds to one region.
    """
    props_table = regionprops_table(img, properties=("label", "centroid"))

    labels = props_table["label"]
    centroid_y = props_table["centroid-0"]  # row
    centroid_x = props_table["centroid-1"]  # col

    # Stack into (x, y) format
    centroids = np.stack([centroid_x, centroid_y], axis=1)

    # Sort by label
    sort_idx = np.argsort(labels)
    sorted_centroids = centroids[sort_idx]
    return sorted_centroids


def get_euclidean_metrics(tform: EuclideanTransform) -> Tuple[float, np.ndarray]:
    """
    Extract the rotation angle (in degrees) and translation vector from
    a 2D EuclideanTransform.

    A Euclidean transform includes:
    - Rotation (around the origin)
    - Translation (shift in x and y)
    - No scaling or shearing

    Parameters
    ----------
    tform : EuclideanTransform
        A skimage.transform.EuclideanTransform object representing a 2D transform.
        The transform matrix is expected to be in homogeneous (3x3) form.

    Returns
    -------
    angle_deg : float
        The rotation angle in degrees. Positive values indicate counter-clockwise rotation.

    translation : np.ndarray of shape (2,)
        The translation vector [tx, ty], representing shifts along the x and y axes (in pixels).
    """

    # Rotation matrix (top-left 2x2)
    R = tform.params[:2, :2]

    # Rotation angle
    angle_rad = np.arctan2(R[0, 1], R[0, 0])
    angle_deg = np.degrees(angle_rad)

    # Translation vector (last column, first two rows)
    translation = tform.params[:2, 2]

    return angle_deg, translation


def transform_euclidean_metric_to_scipy(
    tform: EuclideanTransform,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a EuclideanTransform object from skimage into a format compatible with
    scipy.ndimage.affine_transform (i.e., matrix and offset with (row, col) axis order).

    This function extracts the inverse of the transform and adjusts it to match the
    coordinate conventions used by scipy.ndimage, which expects transforms to be
    applied in (row, col) = (y, x) order, unlike skimage which uses (x, y).

    Parameters
    ----------
    tform : EuclideanTransform
        A skimage.transform.EuclideanTransform object (typically from estimate_transform)
        that defines a 2D Euclidean transformation (rotation + translation).

    Returns
    -------
    matrix : np.ndarray of shape (2, 2)
        The affine transformation matrix in (row, col) axis order, suitable for use
        with scipy.ndimage.affine_transform.

    offset : np.ndarray of shape (2,)
        The translation vector in (row, col) axis order, to be used as the offset
        in scipy.ndimage.affine_transform.
    """
    inverse_matrix = tform.inverse.params  # 3x3 matrix
    # Extract rotation and translation
    matrix = inverse_matrix[:2, :2]
    offset = inverse_matrix[:2, 2]

    # Swap x and y axes to match scipy's (row, col) ordering
    # Swap rows and columns (transpose matrix, flip offset)
    matrix = matrix[[1, 0], :][:, [1, 0]]  # yx -> xy
    offset = offset[[1, 0]]  # yx -> xy

    return matrix, offset


def apply_affine_to_slice(slice_2d, matrix, offset):
    return affine_transform(
        slice_2d, matrix=matrix, offset=offset, order=0, mode="constant", cval=0
    )


def is_identity_transform(tform: EuclideanTransform, rtol=1e-5, atol=1e-8) -> bool:
    """
    Check if a EuclideanTransform is effectively an identity transform (no rotation or translation).

    Parameters
    ----------
    tform : EuclideanTransform
        The transform object from estimate_transform.
    rtol : float
        Relative tolerance for comparing values.
    atol : float
        Absolute tolerance for comparing values.

    Returns
    -------
    bool
        True if the transform is effectively identity; False otherwise.
    """
    # Identity Euclidean matrix
    identity_matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return np.allclose(tform.params, identity_matrix, rtol=rtol, atol=atol)
