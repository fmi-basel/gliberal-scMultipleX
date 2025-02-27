# Copyright (C) 2025 Friedrich Miescher Institute for Biomedical Research

##############################################################################
#                                                                            #
# Author: Nicole Repina              <nicole.repina@fmi.ch>                  #
#                                                                            #
##############################################################################

import logging

import dask.array as da
import numpy as np

from scmultiplex.fractal.fractal_helper_functions import read_table_from_zarrurl

logger = logging.getLogger(__name__)


def load_correction_adata(
    correction_tables_zarr_url, correction_table_name, full_z_count, label_adata
):
    """
    Load z-correction anndata table (with name correction_table_name, string) from
    zarr url (correction_tables_zarr_url, string). Check that correction table has same number of z planes as input
    image data, and that all object labels of the correction table are present in the input segmentation label image.
    Output: anndata object of the z-illumination correction table.
    """
    # Load z-illumination correction table for this channel and perform checks
    correction_adata = read_table_from_zarrurl(
        correction_tables_zarr_url, correction_table_name
    )
    # check that number of entries in anndata matches the z-stack size of image
    adata_z_count = correction_adata.X.shape[1]
    if adata_z_count != full_z_count:
        raise ValueError(
            f"Correction matrix shape does not match image shape. Z-correction table contains "
            f"{adata_z_count} entries, while zarr image has {full_z_count} z-slices."
        )

    correction_labels = correction_adata.obs["label"].to_numpy()
    seg_labels = label_adata.obs["label"].to_numpy()
    if not np.all(np.isin(correction_labels, seg_labels)):
        raise ValueError(
            "Not all objects of illumination correction table are present in the input segmentation "
            "image. Check that task input label_name and roi_table is correct, or that Calculate "
            "Z-Illumination Correction task was run on correct segmentation image."
        )

    return correction_adata


def convert_adata_to_dict_of_lists(adata):
    """
    Convert data from anndata to a dictionary of lists, where:
    # key is obs label (as stored in obs 'label' column, usually a string), e.g. key = '3'.
    # value is list of X values for that obs value, where length of list corresponds to adata.X.shape[1].
    e.g. value = [0.9, 0.9, 1, 0.9, 0.8, 0.7, 0.6]
    Output: dictionary of above key:value pairs.
    """

    adata_dict = {obs: adata.X[i].tolist() for i, obs in enumerate(adata.obs["label"])}

    return adata_dict


def make_correction_block(block1, correction_dict, block_info=None):
    """
    Generate 3D z-illumination correction array from z-illumination correction dictionary. Function to be
    used with Dask map_blocks to parallelize calculation over chunks.
    Input:
    block (i.e. chunk) of dask array,
    correction dictionary: contains z-illumination correction for each object
        (e.g. output of convert_adata_to_dict_of_lists), where key is object label, value is list of correction values
        (typically 0 to 1) over z.
    block_info: passes position information for chunk within full dask array. See
        da.map_blocks documentation for details.
    Output: block where all values are 1, except for regions that contain segmented objects and that have been
        filled in with the z-correction values, based on object id and z-position.
    """

    if block_info is None:
        raise ValueError("Keyword argument block_info is None")

    # extract chunk coordinates from block_info
    chunk_coords = block_info[0]["array-location"]
    # get lower z-index of chunk from full dask array
    z_start = chunk_coords[0][0]

    # return indeces of elements that are non-zero in block as tuple
    pix_nonzero = np.nonzero(block1)

    # initialize new block
    correction_block = np.ones_like(block1, dtype=np.float32)

    # print(list(islice(zip(*pix_nonzero), 10)))
    # todo: could it be faster here to collect all nonzero pixels at given z and label id, and relabel them all at once?
    for nonzero_pixel in zip(*pix_nonzero):
        # nonzero_pixel is [z,y,x]
        # fetch value (as Python scalar) of array at given pixel, as string to match obs
        key = str(int(block1[nonzero_pixel].item()))
        # get z index of the pixel - this is z-index relative to block (e.g. for chunk size 10, ranges 0 to 9)
        z_index = nonzero_pixel[0]
        # get absolute z-index relative to entire dask array
        z_index_abs = z_index + z_start
        # get value of correction from dict, for given z and label
        correction_value = correction_dict[key][z_index_abs]
        # place value into numpy array
        correction_block[tuple(nonzero_pixel)] = correction_value

    return correction_block


def apply_correction(image_dask, correction_dask, background_intensity):
    """
    Apply z-illumination correction to dask array, with parallelization over chunks. Builds dask graph without compute.
    Subtracts background and divides input image by correction, and adds back the background.
    Input:
    image_dask: dask array of channel to be corrected
    correction_dask: dask array of correction values, same shape and chunking as image_dask, e.g. output
        of da.map_blocks(make_correction_block).
    background_intensity: scalar (int) background intensity, user-specified.
    Output: corrected image_dask array.
    """
    dtype = image_dask.dtype
    dtype_max = np.iinfo(dtype).max

    # Set all values <= background_intensity to the background intensity
    # image_dask_nobg = da.where(image_dask > background_intensity, image_dask, background_intensity)
    image_dask_nobg = da.clip(image_dask, background_intensity, None)

    # Subtract background_intensity from all elements
    image_dask_nobg = image_dask_nobg - background_intensity

    # Divide element-wise, this converts array to float
    corrected_image_dask = da.divide(image_dask_nobg, correction_dask)

    # Add back background
    corrected_image_dask = corrected_image_dask + background_intensity

    # Array is automatically upcasted if values higher than dtype_max is generated, here catch any overflow
    if da.max(corrected_image_dask) > dtype_max:
        logger.warning(
            f"Correction generated intensity values beyond the max range of input data type. "
            f"These values have been clipped to {dtype_max=}"
        )

        # Correct clipped values; values above dtype_max are set to dtype_max
        corrected_image_dask = da.clip(corrected_image_dask, None, dtype_max)

    # Convert from float to original dtype (usually int); all floats are floored
    corrected_image_dask = corrected_image_dask.astype(dtype)

    return corrected_image_dask


def run_apply_correction(ch_dask, label_dask, adata, background_intensity):
    """
    Main run function for applying z-iilumination correction to channel images with Dask parallelization over chunks.
    Builds Dask graph, does not compute.
    """
    # make dictionary of values used for correction
    correction_dict = convert_adata_to_dict_of_lists(adata)

    # generate 3d correction array that matches dimensions of label_dask, chunk-wise
    correction_dask = da.map_blocks(
        make_correction_block,
        label_dask,
        correction_dict=correction_dict,
        dtype=np.float32,
    )

    # apply correction to dask array chunk-wise
    corrected_image_dask = apply_correction(
        ch_dask, correction_dask, background_intensity
    )

    return corrected_image_dask
