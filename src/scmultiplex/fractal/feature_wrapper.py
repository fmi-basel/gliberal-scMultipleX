from typing import Any
from typing import Dict
from skimage.measure import regionprops
import numpy as np
import pandas as pd

from scmultiplex.features.FeatureFunctions import (
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
    img: np.array,
    spacing: tuple,
    is_2D: bool,
    measure_morphology=False,
    min_area_fraction=0.005,
    channel_prefix: str = "",
    extra_values: Dict[str, Any] = {},
):
    """
    :param label_img: 2D or 3D numpy array of labeled objects
    :param img: 2D or 3D numpy array of the intensity image to measure
    :param spacing: Tuple of the spacing in z, y, x.
    :param is_2D: Boolean indicating if the image is 2D or 3D
    :param measure_morphology: Boolean indicating if morphology measurements should be made
    :param min_area_fraction: Minimum area fraction for concavity count
    :param channel_prefix: String to prefix to the column names of intensity measurements
    :param extra_values: Dictionary of column names (keys) and constant values
                         (values) for constant values that should be added to
                         each row of the measurement


    :return: Tuple of 2 pandas DataFrames of the measurements and observations
             Observations can contain strings and other non-numeric data
    """

    # Set the global spacing. Is used in surface_area_marchingcube
    set_spacing(spacing)

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
        }
        intensity_measurements_pref = {
            channel_prefix + "." + str(key): val
            for key, val in intensity_measurements.items()
        }

        if measure_morphology:
            morphology_measurements = {
                "imgdim_x": img.shape[-1],
                "imgdim_y": img.shape[-2],
                "is_touching_border_xy": is_touching_border_xy(
                    labeled_obj, img_shape=img.shape
                ),
                "x_pos_weighted_pix": labeled_obj["weighted_centroid"][1],
                "y_pos_weighted_pix": labeled_obj["weighted_centroid"][0],
                "x_massDisp_pix": labeled_obj["weighted_centroid"][1]
                - labeled_obj["centroid"][1],
                "y_massDisp_pix": labeled_obj["weighted_centroid"][0]
                - labeled_obj["centroid"][0],
                "area_bbox": labeled_obj["area_bbox"],
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
                morphology_2D_only = {
                    "x_pos_pix": labeled_obj["centroid"][1],
                    "y_pos_pix": labeled_obj["centroid"][0],
                    "area_pix": labeled_obj["area"],
                    "perimeter": labeled_obj["perimeter"],
                    "concavity": convex_hull_area_resid(labeled_obj),
                    "asymmetry": convex_hull_centroid_dif(labeled_obj),
                    "eccentricity": labeled_obj["eccentricity"],
                    "circularity": circularity(labeled_obj),
                    "concavity_count": concavity_count(
                        labeled_obj, min_area_fraction=min_area_fraction
                    ),
                    "disconnected_components": disconnected_component(
                        labeled_obj.image
                    ),
                }
                morphology_measurements = morphology_measurements | morphology_2D_only
            else:
                morphology_3d_only = {
                    "x_pos_vox": labeled_obj["centroid"][2],
                    "y_pos_vox": labeled_obj["centroid"][1],
                    "z_pos_vox": labeled_obj["centroid"][0],
                    "is_touching_border_z": is_touching_border_z(
                        labeled_obj, img_shape=img.shape
                    ),
                    "area_convhull": labeled_obj["area_convex"],
                    "volume_pix": labeled_obj["area"],
                    "surface_area": labeled_obj["surface_area_marchingcube"],
                }
                morphology_measurements = morphology_measurements | morphology_3d_only

            measurement_rows.append(
                label_info | intensity_measurements_pref | morphology_measurements
            )
        else:
            measurement_rows.append(label_info | intensity_measurements_pref)

    df_well = pd.DataFrame(measurement_rows)
    df_obs = pd.DataFrame(observation_rows)

    return df_well, df_obs
