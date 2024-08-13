# Copyright (C) 2024 Friedrich Miescher Institute for Biomedical Research

##############################################################################
#                                                                            #
# Author: Nicole Repina              <nicole.repina@fmi.ch>                  #
#                                                                            #
##############################################################################
import numpy as np
import pandas as pd
import math

from fractal_tasks_core.roi import load_region, convert_indices_to_regions
from fractal_tasks_core.upscale_array import upscale_array
from skimage.measure import regionprops_table

from scmultiplex.linking.NucleiLinkingFunctions import calculate_quantiles

# warnings.filterwarnings("ignore")


def filter_small_sizes_per_round(props_numpy, column=-1, threshold=0.05):
    """
    props_np: numpy array, where each row corresponds to a nucleus and columns are regionprops measurements
    Filter out nuclei with small volumes, presumed to be debris.
    Column: index of column in numpy array that correspond to volume measurement
    Threshold: multiplier that specifies cutoff for volumes below which nuclei are filtered out, float in range [0,1],
        e.g. 0.05 means that 5% of median of nuclear volume distribution is used as cutoff.
    Return filtered numpy array (moving_pc_filtered, fixed_pc_filtered) as well as logging metrics.
    """
    median, _ = calculate_quantiles(props_numpy, q=0.5, column=column)

    # generate boolean arrays for filtering
    volume_cutoff = threshold * median
    row_selection = props_numpy[:, column].transpose() > volume_cutoff

    # filter pc to remove small nuclei
    props_numpy_filtered = props_numpy[row_selection, :]

    # return nuclear ids of removed nuclei, assume nuclear labels are first column (column 0)
    removed = props_numpy[~row_selection, 0]

    # calculate mean of removed nuclear sizes
    if props_numpy[~row_selection, column].size == 0:
        removed_size_mean = None
    else:
        removed_size_mean = np.mean(props_numpy[~row_selection, column])

    # calculate mean of kept nuclear sizes
    size_mean = np.mean(props_numpy_filtered[:, column])

    return props_numpy_filtered, removed, removed_size_mean, size_mean, volume_cutoff


def calculate_mean_volume(seg):
    """
    Calculate mean volume across labels in input numpy array segmentation image
    """
    props = regionprops_table(label_image=seg, properties=('label', 'area'))

    # output column order must be: ["label", "volume"]
    # units are in pixels
    props = (pd.DataFrame(props, columns=['label', 'area'])).to_numpy()

    # calculate mean of nuclear volumes
    size_mean = np.mean(props[:, -1])

    return size_mean


def mask_by_parent_object(seg, parent_dask, parent_idlist_parentmeta, row_int, parent_label_id):
    """
    Mask input numpy array (seg) by parent numpy array (region loaded from dask_parent)
    Assumes seg array already loaded into memory, and here load matching organoid array to perform multiplication
    Mask is the parent object loaded from parent zarr array, where
        parent_idlist_parentmeta is output of convert_ROI_table_to_indices function
    e.g.
        parent_adata = ad.read_zarr(parent_roi_path)
        parent_dask = da.from_zarr(parent_label_path)

        # Read Zarr metadata
        parent_ngffmeta = load_NgffImageMeta(f"{input_zarr_path}/labels/{label_name_parent}")
        parent_xycoars = parent_ngffmeta.coarsening_xy
        parent_pixmeta = parent_ngffmeta.get_pixel_sizes_zyx(level=level)

        parent_idlist_parentmeta = convert_ROI_table_to_indices(
            parent_adata,
            level=level,
            coarsening_xy=parent_xycoars,
            full_res_pxl_sizes_zyx=parent_pixmeta,
        )
        check_valid_ROI_indices(parent_idlist_parentmeta, roi_table_parent)

    row_int is the row index (integer) of region to load
    e.g.
        parent_labels = parent_adata.obs_vector('label')
        for row in parent_adata.obs_names:
            row_int = int(row)
    parent_label_id is the label id of organoid
    e.g.
        parent_label_id = parent_labels[row_int]

    return numpy array output of nuc or cell segmentation masked by organoid seg
    """
    # load object label image for object in reference round
    parent = load_region(
        data_zyx=parent_dask,
        region=convert_indices_to_regions(parent_idlist_parentmeta[row_int]),
        compute=True,
    )
    # if object segmentation was run at a different level than nuclear segmentation,
    # need to upscale arrays to match shape
    if parent.shape != seg.shape:
        parent = upscale_array(array=parent, target_shape=seg.shape, pad_with_zeros=False)

    # mask nuclei by parent object
    parent_mask = np.zeros_like(parent)
    parent_mask[parent == int(parent_label_id)] = 1  # select only current object and binarize object mask
    seg_masked = seg * parent_mask

    return seg_masked


def load_border_values(nparray):
    """
    Load edge (border) values of a 2-dim numpy array
    Return: 1-d numpy array of edge values
    """
    if len(nparray.shape) != 2:
        raise ValueError("Expecting 2-dimensional image (z-slice) as input")
    alist = [nparray[0, :], nparray[1:, -1], nparray[-1, :-1], nparray[1:-1, 0]]
    edge_values = np.concatenate(alist)
    return edge_values


def remove_border(nparray):
    """
    Remove edge (border) values from a 2-dim numpy array, e.g. to reverse padding
    Return: 2-d numpy array with edge values removed
    """
    if len(nparray.shape) != 2:
        raise ValueError("Expecting 2-dimensional image (z-slice) as input")
    return nparray[1:-1, 1:-1]
