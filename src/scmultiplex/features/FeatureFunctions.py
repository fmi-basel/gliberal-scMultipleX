import numpy as np
import scipy


def fixed_percentiles(region_mask, intensity):
    """
    Calculate values at percentiles 25, 50, 75, 90, 95, 99 of foreground pixels of intensity image
    :param region_mask: binary foreground background mask
    :param intensity: intensity image
    :return: intensity value list[float]
    """
    assert region_mask.dtype == bool, "Region mask must be bool type"
    return np.percentile(intensity[region_mask], q=(25, 50, 75, 90, 95, 99))


def skewness(region_mask, intensity):
    return scipy.stats.skew(intensity[region_mask])


def kurtos(region_mask, intensity):
    return scipy.stats.kurtosis(intensity[region_mask])


def stdv(region_mask, intensity):
    # ddof=1 for sample var
    return np.std(intensity[region_mask], ddof=1)
