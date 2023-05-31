# Copyright (C) 2023 Friedrich Miescher Institute for Biomedical Research

##############################################################################
#                                                                            #
# Author: Nicole Repina              <nicole.repina@fmi.ch>                  #
# Author: Enrico Tagliavini          <enrico.tagliavini@fmi.ch>              #
#                                                                            #
##############################################################################

import numpy as np
import pandas as pd
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift
from scmultiplex.linking.matching import matching


def pad_img_set(img1, img2):
    """
    Pad two images (numpy arrays) to be the same size
    :param img1: typically R0
    :param img2: typically RX
    :return: two padded images
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
    :param img: numpy array
    :return: binarized image (0 or 1)
    """
    img = img.copy() # copy to not change original variable
    img[img > 0] = 1
    return img


def subsample_image(img, bin):
    """
    Subsample 2D image by indicated bin spacing.
    :param img: 2D numpy array image
    :bin: integer value for subsampling, ex. bin = 4 means every 4th pixel is sampled
    :return: subsampled image with image dimensions reduced by bin^2
    """
    img_bin = img[::bin, ::bin]
    return img_bin


def calculate_shift(img0, imgX, bin):
    """
    Calculate xy shift between a set of 2D images based on phase cross correlation
    :param img0: 2D numpy array image, reference image ex. R0
    :param imgX: 2D numpy array image, moving image ex. RX
    :bin: integer value for subsampling, ex. bin = 4 means every 4th pixel is sampled
    :return:
        shifts: ndarray shift vector (y,x) in pixels required to shift moving image (imgX) relative to reference image (img0)
        pixel shift is relative to input imgX scaling (i.e. before subsampling)
        img0_pad: 2D numpy array image, padded to be same size as imgX_pad
        imgX_pad: 2D numpy array image, padded to be same size as img0_pad
    """

    # pad so that images have same shape
    img0_pad, imgX_pad = pad_img_set(img0, imgX)

    if (img0_pad.shape[0] != imgX_pad.shape[0]) | (img0_pad.shape[1] != imgX_pad.shape[1]):
        raise ValueError("image pair must have same dimensions")

    # binarize padded overviews
    img0_pad_binary, imgX_pad_binary = binarize_img(img0_pad), binarize_img(imgX_pad)

    # subsample so that registration runs faster
    img0_pad_binary_bin, imgX_pad_binary_bin = subsample_image(img0_pad_binary, bin=bin), subsample_image(imgX_pad_binary, bin=bin)

    # calculate shifts and take into account subsampling
    result = phase_cross_correlation(img0_pad_binary_bin, imgX_pad_binary_bin)
    shifts = bin * result[0]  # (y,x)

    return shifts, img0_pad, imgX_pad


def apply_shift(img, shifts):
    """
    Apply shift from image registration to image
    :param img: 2D numpy array image
    :shifts: ndarray shift vector (y,x) in pixels
    :return: shifted 2D numpy array image
    """

    img_shifted = shift(img, shifts, mode="constant", cval=0)

    return img_shifted


def calculate_matching(img0, imgX, iou_cutoff):
    stat = matching(img0, imgX, criterion="iou", thresh=iou_cutoff, report_matches=True)

    df = pd.DataFrame(
        list(zip([x[0] for x in stat[14]], [x[1] for x in stat[14]], stat[15])),
        columns=["R0_label", "RX_label", "iou"],
    )
    df_filt = df[df["iou"] > iou_cutoff]

    return stat, df, df_filt








