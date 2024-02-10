# Copyright (C) 2024 Friedrich Miescher Institute for Biomedical Research

##############################################################################
#                                                                            #
# Author: Nicole Repina              <nicole.repina@fmi.ch>                  #
#                                                                            #
##############################################################################

import warnings

import numpy as np
import math

from scmultiplex.linking.NucleiLinkingFunctions import calculate_quantiles

warnings.filterwarnings("ignore")


# TODO refactor calculate_platymatch_registration to also use this function instead of filter_small_sizes!
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
    row_selection = props_numpy[:, column].transpose() > (threshold * median)

    # filter pc to remove small nuclei
    props_numpy_filtered = props_numpy[row_selection, :]

    # return nuclear ids of removed nuclei, assume nuclear labels are first column (column 0)
    removed = props_numpy[~row_selection, 0]

    # calculate mean of removed nuclear sizes
    removed_size_mean = np.mean(props_numpy[~row_selection, column])

    # calculate mean of kept nuclear sizes
    size_mean = np.mean(props_numpy_filtered[:, column])

    return props_numpy_filtered, removed, removed_size_mean, size_mean


def equivalent_diam(volume):
    """
    Calculate equivalent diameter of sphere with a given volume
    """
    try:
        diameter = 2 * math.pow((0.75 * (1/np.pi) * volume), (1/3))
    # ValueError is raised when input value is negative
    # in this case set diameter to 0
    except ValueError:
        diameter = 0
    return diameter



