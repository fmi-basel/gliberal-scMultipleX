# Copyright (C) 2023 Friedrich Miescher Institute for Biomedical Research

##############################################################################
#                                                                            #
# Author: Nicole Repina              <nicole.repina@fmi.ch>                  #
# Author: Tim-Oliver Buchholz        <tim-oliver.buchholz@fmi.ch>            #
# Author: Enrico Tagliavini          <enrico.tagliavini@fmi.ch>              #
#                                                                            #
##############################################################################

import pandas as pd

from scmultiplex.features.feature_wrapper import get_regionprops_measurements
from scmultiplex.utils.save_utils import save_to_record

object_types = ["ovr", "org", "nuc", "mem"]


def measure_features(
    object_type,
    record,
    channel,
    label_img,
    img,
    spacing,
    is_2D,
    measure_morphology,
    min_area_fraction,
    channel_prefix,
    extra_values_common,
    extra_values_object,
    touching_labels,
):

    if object_type not in object_types:
        raise ValueError("object type must be one of: %s" % ", ".join(object_types))

    # if user inputs 3D spacing, crop to 2D for organoid feature extraction
    if object_type == "org" and len(spacing) == 3:
        spacing = spacing[1:]

    extra_values = extra_values_common.copy()
    extra_values.update(extra_values_object)

    num_df, info_df = get_regionprops_measurements(
        label_img=label_img,
        img=img,
        spacing=spacing,
        is_2D=is_2D,
        measure_morphology=measure_morphology,
        min_area_fraction=min_area_fraction,
        channel_prefix=channel_prefix,
        extra_values=extra_values,
    )

    num_df = num_df.drop("label", axis=1)
    if object_type == "org":
        info_df = info_df.drop("label", axis=1)
    elif object_type == "ovr":
        info_df = info_df.rename(columns={"label": "org_id"})
        info_df["flag_tile_border"] = info_df["org_id"].isin(touching_labels)

        num_df = num_df.rename(
            columns={"x_pos_pix": "x_pos_pix_global", "y_pos_pix": "y_pos_pix_global"}
        )
    else:
        info_df = info_df.rename(columns={"label": object_type + "_id"})

    df = pd.concat([info_df, num_df], axis=1)

    # Save measurement into the organoid directory.
    name = "regionprops_" + object_type + "_" + str(channel)
    save_to_record(record, name, df)

    return
