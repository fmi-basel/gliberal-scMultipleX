from typing import Any, Dict, Union

import numpy as np
import pandas as pd
from skimage.measure import regionprops

from scmultiplex.config import spacing_anisotropy_scalar, spacing_to2d
from scmultiplex.features.FeatureFunctions import (
    aspect_ratio,
    centroid_weighted_correct,
    circularity,
    convex_hull_area_resid,
    convex_hull_centroid_dif,
    disconnected_component,
    fixed_percentiles,
    is_touching_border_xy,
    is_touching_border_z,
    kurtos,
    minor_major_axis_ratio,
    set_spacing,
    skewness,
    stdv,
    surface_area_marchingcube,
)


def get_regionprops_measurements(
    label_img: np.array,
    img: Union[np.array, None],
    spacing: Union[tuple, None],
    is_2D: bool,
    measure_morphology=False,
    measure_surface_area=False,
    min_area_fraction=0.005,
    channel_prefix: Union[str, None] = None,
    extra_values: Union[Dict[str, Any], None] = None,
):
    """
    :param label_img: 2D or 3D numpy array of labeled objects
    :param img: Optional 2D or 3D numpy array of the intensity image to measure.
                If None, only the label image is measured.
    :param spacing: Tuple of the spacing in z, y, x.
    :param is_2D: Boolean indicating if the image is 2D or 3D
    :param measure_morphology: Boolean indicating if morphology measurements should be made
    :param measure_surface_area: Boolean indicating if surface area measurements should be made
    :param min_area_fraction: Minimum area fraction for concavity count
    :param channel_prefix: String to prefix to the column names of
                           intensity measurements. Defaults to None => no prefix
    :param extra_values: Dictionary of column names (keys) and constant values
                         (values) for constant values that should be added to
                         each row of the measurement


    :return: Tuple of 2 pandas DataFrames of the measurements and observations
             Observations can contain strings and other non-numeric data
    """

    # Set the global spacing. Is used in surface_area_marchingcube
    set_spacing(spacing)

    if extra_values is None:
        extra_values = {}

    extra_properties = [
        fixed_percentiles,
        skewness,
        kurtos,
        stdv,
    ]

    if measure_surface_area:
        extra_properties.append(surface_area_marchingcube)

    regionprops_kwargs = {
        "extra_properties": tuple(extra_properties),
        "spacing": spacing,
    }

    if img is None:
        regionproperties = regionprops(label_img, **regionprops_kwargs)
    else:
        regionproperties = regionprops(label_img, img, **regionprops_kwargs)

    measurement_rows = []
    observation_rows = []

    for labeled_obj in regionproperties:
        obs_info = {
            "label": int(labeled_obj["label"]),
        }

        # Add extra values to the obs_info
        for extra_key, extra_val in extra_values.items():
            obs_info[extra_key] = extra_val

        observation_rows.append(obs_info)

        label_info = {
            # Always include the labels in the core measurements to allow
            # merging with df_obs
            "label": int(labeled_obj["label"]),
        }

        # Always include coordinates
        coordinate_measurements = get_coordinates(labeled_obj, spacing, is_2D)
        label_info.update(coordinate_measurements)

        if measure_morphology:
            morphology_measurements = get_morphology_measurements(
                labeled_obj,
                label_img.shape,
                spacing,
                is_2D,
                min_area_fraction,
                measure_surface_area,
            )

            label_info.update(morphology_measurements)

        if img is not None:
            intensity_measurements = get_intensity_measurements(
                labeled_obj, channel_prefix, spacing, is_2D
            )

            label_info.update(intensity_measurements)

        measurement_rows.append(label_info)

    df_well = pd.DataFrame(measurement_rows)
    df_obs = pd.DataFrame(observation_rows)

    return df_well, df_obs


def get_regionprops_pixel_measurements(
    label_img: np.array,
    img: np.array,
    intensity_threshold: int,
    calculate_area: bool,
    is_2D: bool,
    spacing: Union[tuple, None],
    channel_prefix: Union[str, None] = None,
    extra_values: Dict[str, Any] = {},
):
    """
    Measure per-object pixel-based properties from a labeled image and an
    associated intensity image.

    For each labeled object, computes spatial coordinates and the number of
    pixels whose intensity exceeds a specified threshold. Optionally includes
    the total number of pixels in the object. Measurements are returned as a
    DataFrame, along with a separate observation DataFrame containing labels
    and any additional metadata.

    :param label_img: 2D or 3D numpy array containing labeled objects where
                      each unique nonzero integer represents one object.

    :param img: 2D or 3D numpy array containing the intensity image used for
                threshold-based measurements. Must have the same dimensions
                as ``label_img``.

    :param intensity_threshold: Intensity threshold used to count pixels
                                within each object. Pixels with intensity
                                greater than this value are included.

    :param calculate_area: Boolean indicating whether to include the total
                           number of pixels/voxels in each labeled object
                           (``n_pixels_total``).

    :param is_2D: Boolean indicating whether the input image is 2D or 3D.

    :param spacing: Tuple describing voxel spacing in (z, y, x) order for
                    3D images, or corresponding spatial dimensions for 2D
                    images. Used for coordinate calculations.

    :param channel_prefix: Optional string prefix added to measurement column
                           names. Useful when combining measurements from
                           multiple image channels. Defaults to None.

    :param extra_values: Dictionary of column names (keys) and constant values
                         (values) that will be added to each row in the
                         observations DataFrame.

    :return: Tuple of two pandas DataFrames:

             - df_well: Numeric measurements for each labeled object,
               including coordinates, threshold-based pixel counts, and
               optionally object size.
             - df_obs: Observation metadata for each object, including labels
               and any additional user-provided values.
    """

    regionproperties = regionprops(
        label_img,
        img,
        spacing=spacing,
    )

    measurement_rows = []
    observation_rows = []

    for labeled_obj in regionproperties:
        obs_info = {
            "label": int(labeled_obj["label"]),
        }

        # Add extra values to the obs_info
        for extra_key, extra_val in extra_values.items():
            obs_info[extra_key] = extra_val

        observation_rows.append(obs_info)

        label_info = {
            # Always include the labels in the core measurements to allow
            # merging with df_obs
            "label": int(labeled_obj["label"]),
        }

        # Always include coordinates
        coordinate_measurements = get_coordinates(labeled_obj, spacing, is_2D)
        label_info.update(coordinate_measurements)

        # Measure area (number of pixels) in object, only for first channel
        if calculate_area:
            area_measurements = {
                "n_pixels_total": int(labeled_obj.num_pixels),
            }
            label_info.update(area_measurements)

        # Measure number pixels above threshold
        intensity_measurements = get_pixel_measurements(
            labeled_obj, channel_prefix, intensity_threshold
        )

        label_info.update(intensity_measurements)

        measurement_rows.append(label_info)

    df_well = pd.DataFrame(measurement_rows)
    df_obs = pd.DataFrame(observation_rows)

    return df_well, df_obs


def get_intensity_measurements(labeled_obj, channel_prefix, spacing, is_2D):
    intensity_measurements = {
        "intensity_mean": labeled_obj["intensity_mean"],
        "intensity_max": labeled_obj["intensity_max"],
        "intensity_min": labeled_obj["intensity_min"],
        "percentile25": labeled_obj["fixed_percentiles"][0],
        "percentile50": labeled_obj["fixed_percentiles"][1],
        "percentile75": labeled_obj["fixed_percentiles"][2],
        "percentile90": labeled_obj["fixed_percentiles"][3],
        "percentile95": labeled_obj["fixed_percentiles"][4],
        "percentile99": labeled_obj["fixed_percentiles"][5],
        "stdev": labeled_obj["stdv"],
        "skew": labeled_obj["skewness"],
        "kurtosis": labeled_obj["kurtos"],
    }

    # workaround for buggy version of skimage 0.20.0 . False if we don't need to apply the workaround
    if False:
        corrected_weighted_centroid = centroid_weighted_correct(labeled_obj)
    else:
        corrected_weighted_centroid = labeled_obj
    corrected_weighted_centroid = centroid_weighted_correct(labeled_obj)
    intensity_measurements["x_pos_weighted"] = corrected_weighted_centroid[-1]
    intensity_measurements["y_pos_weighted"] = corrected_weighted_centroid[-2]
    intensity_measurements["x_massDisp"] = (
        corrected_weighted_centroid[-1] - labeled_obj["centroid"][-1]
    )
    intensity_measurements["y_massDisp"] = (
        corrected_weighted_centroid[-2] - labeled_obj["centroid"][-2]
    )
    if not is_2D:
        intensity_measurements["z_pos_weighted"] = corrected_weighted_centroid[-3]
        intensity_measurements["z_massDisp"] = (
            corrected_weighted_centroid[-3] - labeled_obj["centroid"][-3]
        )

    # channel prefix addition is optional
    if channel_prefix is not None:
        intensity_measurements_pref = {
            channel_prefix + "." + str(key): val
            for key, val in intensity_measurements.items()
        }
    else:
        intensity_measurements_pref = intensity_measurements

    return intensity_measurements_pref


def get_pixel_measurements(
    labeled_obj,
    channel_prefix,
    intensity_threshold,
):
    # Load pixels belonging to this label only
    region_intensities = labeled_obj.intensity_image[labeled_obj.image]

    # Count number of pixels above this threshold
    n_pixels_above_threshold = np.sum(region_intensities > intensity_threshold)

    intensity_measurements = {
        "n_pixels_above_threshold": int(n_pixels_above_threshold),
        "intensity_threshold": intensity_threshold,
    }

    # channel prefix addition is optional
    if channel_prefix is not None:
        intensity_measurements_pref = {
            channel_prefix + "." + str(key): val
            for key, val in intensity_measurements.items()
        }
    else:
        intensity_measurements_pref = intensity_measurements

    return intensity_measurements_pref


def get_morphology_measurements(
    labeled_obj,
    img_shape,
    spacing,
    is_2D,
    min_area_fraction,
    measure_surface_area,
):
    morphology_measurements = {
        "is_touching_border_xy": is_touching_border_xy(
            labeled_obj, img_shape=img_shape
        ),
        "imgdim_x": img_shape[-1],
        "imgdim_y": img_shape[-2],
        "area_bbox": labeled_obj["area_bbox"],
        "area_convhull": labeled_obj["area_convex"],
        "equivDiam": labeled_obj["equivalent_diameter_area"],
        "extent": labeled_obj["extent"],
        "solidity": labeled_obj["solidity"],
    }
    # Sometimes, major & minor axis calculations fail with a
    # ValueError: math domain error
    try:
        morphology_measurements["axis_major_length"] = labeled_obj["axis_major_length"]
        morphology_measurements["axis_minor_length"] = labeled_obj["axis_minor_length"]
        morphology_measurements["minmajAxisRatio"] = minor_major_axis_ratio(labeled_obj)
        morphology_measurements["aspectRatio_equivalentDiameter"] = aspect_ratio(
            labeled_obj
        )
    except ValueError:
        morphology_measurements["majorAxisLength"] = np.NaN
        morphology_measurements["minorAxisLength"] = np.NaN
        morphology_measurements["minmajAxisRatio"] = np.NaN
        morphology_measurements["aspectRatio_equivalentDiameter"] = np.NaN

    if is_2D:
        spacing_2d = spacing_to2d(spacing)
        morphology_2D_only = {
            "area": labeled_obj["area"],
            "perimeter": labeled_obj["perimeter"],
            "concavity": convex_hull_area_resid(labeled_obj),
            "asymmetry": convex_hull_centroid_dif(labeled_obj, spacing_2d),
            "eccentricity": labeled_obj["eccentricity"],
            "circularity": circularity(labeled_obj),
            "disconnected_components": disconnected_component(labeled_obj.image),
        }
        morphology_measurements.update(morphology_2D_only)
    else:
        morphology_3d_only = {
            "imgdim_z": img_shape[-3],
            "is_touching_border_z": is_touching_border_z(
                labeled_obj, img_shape=img_shape
            ),
        }

        if measure_surface_area:
            morphology_3d_only["surface_area"] = labeled_obj[
                "surface_area_marchingcube"
            ]

        morphology_measurements.update(morphology_3d_only)

    return morphology_measurements


def get_coordinates(labeled_obj, spacing, is_2D):
    coordinate_measurements = {
        "x_pos": labeled_obj["centroid"][-1],
        "y_pos": labeled_obj["centroid"][-2],
    }

    if not is_2D:
        coordinate_measurements_3D = {
            "z_pos": labeled_obj["centroid"][-3],
            "z_pos_pix": labeled_obj["centroid"][-3]
            / spacing_anisotropy_scalar(spacing),
            "volume": labeled_obj["area"],
        }
        coordinate_measurements.update(coordinate_measurements_3D)

    return coordinate_measurements
