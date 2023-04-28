from typing import Any, Dict, Union
from skimage.measure import regionprops
import numpy as np
import pandas as pd

from scmultiplex.config import spacing_anisotropy_scalar, spacing_to2d
from scmultiplex.features.FeatureFunctions import (
    centroid_weighted_correct,
    fixed_percentiles,
    kurtos,
    skewness,
    stdv,
    disconnected_component,
    surface_area_marchingcube,
    is_touching_border_xy,
    is_touching_border_z,
)

from scmultiplex.features.FeatureFunctions import (
    minor_major_axis_ratio,
    convex_hull_area_resid,
    convex_hull_centroid_dif,
    circularity,
    aspect_ratio,
    concavity_count,
    set_spacing,
)


def get_regionprops_measurements(
    label_img: np.array,
    img: Union[np.array, None],
    spacing: Union[tuple, None],
    is_2D: bool,
    measure_morphology=False,
    min_area_fraction=0.005,
    channel_prefix: Union[str, None] = None,
    extra_values: Dict[str, Any] = {},
):
    """
    :param label_img: 2D or 3D numpy array of labeled objects
    :param img: Optional 2D or 3D numpy array of the intensity image to measure.
                If None, only the label image is measured.
    :param spacing: Tuple of the spacing in z, y, x.
    :param is_2D: Boolean indicating if the image is 2D or 3D
    :param measure_morphology: Boolean indicating if morphology measurements should be made
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

    if img is None:
        regionproperties = regionprops(
            label_img,
            extra_properties=(
                fixed_percentiles,
                skewness,
                kurtos,
                stdv,
                surface_area_marchingcube,
            ),
            spacing=spacing,
        )
    else:
        regionproperties = regionprops(
            label_img,
            img,
            extra_properties=(
                fixed_percentiles,
                skewness,
                kurtos,
                stdv,
                surface_area_marchingcube,
            ),
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

        if measure_morphology:
            morphology_measurements = get_morphology_measurements(
                labeled_obj, label_img.shape, spacing, is_2D, min_area_fraction
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


def get_intensity_measurements(labeled_obj, channel_prefix, spacing, is_2D):
    intensity_measurements = {
        "mean_intensity": labeled_obj["mean_intensity"],
        "max_intensity": labeled_obj["max_intensity"],
        "min_intensity": labeled_obj["min_intensity"],
        "percentile25": labeled_obj["fixed_percentiles"][0],
        "percentile50": labeled_obj["fixed_percentiles"][1],
        "percentile75": labeled_obj["fixed_percentiles"][2],
        "percentile90": labeled_obj["fixed_percentiles"][3],
        "percentile95": labeled_obj["fixed_percentiles"][4],
        "percentile99": labeled_obj["fixed_percentiles"][5],
        "stdev": labeled_obj["stdv"],
        "skew": labeled_obj["skewness"],
        "kurtosis": labeled_obj["kurtos"],
        # "x_pos_weighted_pix": labeled_obj["weighted_centroid"][-1],
        # "y_pos_weighted_pix": labeled_obj["weighted_centroid"][-2],
        # "x_massDisp_pix": labeled_obj["weighted_centroid"][-1]
        # - labeled_obj["centroid"][-1],
        # "y_massDisp_pix": labeled_obj["weighted_centroid"][-2]
        # - labeled_obj["centroid"][-2],
    }

    # add 3D-specific intensity measurements
    # if not is_2D:
    #     intensity_3D_only = {
    #         "z_pos_weighted_pix": labeled_obj["weighted_centroid"][-3],
    #         "z_massDisp_pix": labeled_obj["weighted_centroid"][-3]
    #         - labeled_obj["centroid"][-3],
    #     }
    #     intensity_measurements.update(intensity_3D_only)

    # New centroid weighting block
    if True:
        corrected_weighted_centroid = centroid_weighted_correct(labeled_obj)
        intensity_measurements['x_pos_weighted_pix'] = corrected_weighted_centroid[-1]
        intensity_measurements['y_pos_weighted_pix'] = corrected_weighted_centroid[-2]
        intensity_measurements['x_massDisp_pix'] = corrected_weighted_centroid[-1] - labeled_obj["centroid"][-1]
        intensity_measurements['y_massDisp_pix'] = corrected_weighted_centroid[-2] - labeled_obj["centroid"][-2]
        if not is_2D:
            intensity_measurements['z_pos_weighted_pix'] = corrected_weighted_centroid[-3]
            intensity_measurements['z_massDisp_pix'] = corrected_weighted_centroid[-3] - labeled_obj["centroid"][-3]

    # channel prefix addition is optional
    if channel_prefix is not None:
        intensity_measurements_pref = {
            channel_prefix + "." + str(key): val
            for key, val in intensity_measurements.items()
        }
    else:
        intensity_measurements_pref = intensity_measurements
    
    return intensity_measurements_pref


def get_morphology_measurements(labeled_obj, img_shape, spacing, is_2D, min_area_fraction):
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
        morphology_measurements["majorAxisLength"] = labeled_obj[
            "major_axis_length"
        ]
        morphology_measurements["minorAxisLength"] = labeled_obj[
            "minor_axis_length"
        ]
        morphology_measurements["minmajAxisRatio"] = minor_major_axis_ratio(
            labeled_obj
        )
        morphology_measurements[
            "aspectRatio_equivalentDiameter"
        ] = aspect_ratio(labeled_obj)
    except ValueError:
        morphology_measurements["majorAxisLength"] = np.NaN
        morphology_measurements["minorAxisLength"] = np.NaN
        morphology_measurements["minmajAxisRatio"] = np.NaN
        morphology_measurements["aspectRatio_equivalentDiameter"] = np.NaN

    if is_2D:
        spacing_2d = spacing_to2d(spacing)
        morphology_2D_only = {
            "area_pix": labeled_obj["area"],
            "perimeter": labeled_obj["perimeter"],
            "concavity": convex_hull_area_resid(labeled_obj),
            "asymmetry": convex_hull_centroid_dif(labeled_obj, spacing_2d),
            "eccentricity": labeled_obj["eccentricity"],
            "circularity": circularity(labeled_obj),
            "concavity_count": concavity_count(
                labeled_obj, min_area_fraction=min_area_fraction
            ),
            "disconnected_components": disconnected_component(
                labeled_obj.image
            ),
        }
        morphology_measurements.update(morphology_2D_only)
    else:
        morphology_3d_only = {
            "imgdim_z": img_shape[-3],
            "is_touching_border_z": is_touching_border_z(
                labeled_obj, img_shape=img_shape
            ),
            "volume_pix": labeled_obj["area"],
            "surface_area": labeled_obj["surface_area_marchingcube"],
        }
        morphology_measurements.update(morphology_3d_only)

    return morphology_measurements

def get_coordinates(labeled_obj, spacing, is_2D):
    coordinate_measurements = {
        "x_pos_pix": labeled_obj["centroid"][-1],
        "y_pos_pix": labeled_obj["centroid"][-2],
    }

    if not is_2D:
        coordinate_measurements_3D = {
            "z_pos_pix_scaled": labeled_obj["centroid"][-3],
            "z_pos_pix_img": labeled_obj["centroid"][-3]
            / spacing_anisotropy_scalar(spacing)
        }
        coordinate_measurements.update(coordinate_measurements_3D)

    return coordinate_measurements
