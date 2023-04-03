# Copyright (C) 2023 Friedrich Miescher Institute for Biomedical Research

##############################################################################
#                                                                            #
# Author: Nicole Repina              <nicole.repina@fmi.ch>                  #
# Author: Tim-Oliver Buchholz        <tim-oliver.buchholz@fmi.ch>            #
# Author: Enrico Tagliavini          <enrico.tagliavini@fmi.ch>              #
#                                                                            #
##############################################################################

import os
import pandas as pd

from traceback import print_exc

from scmultiplex.features.FeatureFunctions import (
    minor_major_axis_ratio,
    convex_hull_area_resid,
    convex_hull_centroid_dif,
    circularity,
    aspect_ratio,
    concavity_count
)

from scmultiplex.fractal.feature_wrapper import get_regionprops_measurements

object_types = ['org', 'nuc', 'mem']

def measure_features(object_type,
                     organoid,
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
                     ):

    if object_type not in object_types:
        raise ValueError('object type must be one of: %s' % ', '.join(object_types))

    # if user inputs 3D spacing, crop to 2D for organoid feature extraction
    if object_type == 'org' and len(spacing) == 3:
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

    num_df = num_df.drop('label', axis=1)
    if object_type == 'org':
        info_df = info_df.drop('label', axis=1)
    else:
        info_df = info_df.rename(columns={"label": object_type + "_id"})

    df = pd.concat([info_df, num_df], axis=1)

    # Save measurement into the organoid directory.
    name = "regionprops_" + object_type + "_" + str(channel)
    path = os.path.join(organoid.organoid_dir, name + ".csv")
    df.to_csv(path, index=False)  # saves csv

    # Add the measurement to the faim-hcs datastructure and save.
    organoid.add_measurement(name, path)
    organoid.save()  # updates json file

    return


def regionprops_to_row(ot, regionproperties, org_channel, nuc_channel, mem_channel, organoid, channel, mask_ending, nuc_ending,
                       mem_ending, abs_min_intensity=None, img_dim=None, disconnected=None):
    if ot not in object_types:
        raise ValueError('object type must be one of: %s' % ', '.join(object_types))

    row_list = []

    for labeled_obj in regionproperties:

        common_row = {
            'hcs_experiment': organoid.well.plate.experiment.name,
            'root_dir': organoid.well.plate.experiment.root_dir,
            'plate_id': organoid.well.plate.plate_id,
            'well_id': organoid.well.well_id,
            'channel_id': channel,
            'object_type': ot,
            'org_id': int(organoid.organoid_id.rpartition("_")[2]),
            'intensity_img': organoid.raw_files[channel],
            'mean_intensity': labeled_obj['mean_intensity'],
            'max_intensity': labeled_obj['max_intensity'],
            'min_intensity': labeled_obj['min_intensity'],
            'percentile25': labeled_obj['fixed_percentiles'][0],
            'percentile50': labeled_obj['fixed_percentiles'][1],
            'percentile75': labeled_obj['fixed_percentiles'][2],
            'percentile90': labeled_obj['fixed_percentiles'][3],
            'percentile95': labeled_obj['fixed_percentiles'][4],
            'percentile99': labeled_obj['fixed_percentiles'][5],
            'stdev': labeled_obj['stdv'],
            'skew': labeled_obj['skewness'],
            'kurtosis': labeled_obj['kurtos']
        }

        try:
            if ot == 'organoid' and channel == org_channel:
                # include both channel and shape features
                row = {
                    'segmentation_org': organoid.segmentations[mask_ending],
                    'imgdim_x': img_dim[1],
                    'imgdim_y': img_dim[0],
                    'x_pos_pix': labeled_obj['centroid'][1],
                    'y_pos_pix': labeled_obj['centroid'][0],
                    'x_pos_weighted_pix': labeled_obj['weighted_centroid'][1],
                    'y_pos_weighted_pix': labeled_obj['weighted_centroid'][0],
                    'x_massDisp_pix': labeled_obj['weighted_centroid'][1] - labeled_obj['centroid'][1],
                    'y_massDisp_pix': labeled_obj['weighted_centroid'][0] - labeled_obj['centroid'][0],
                    'abs_min': abs_min_intensity,
                    'area_pix': labeled_obj['area'],
                    'area_convhull': labeled_obj['area_convex'],
                    'area_bbox': labeled_obj['area_bbox'],
                    'perimeter': labeled_obj['perimeter'],
                    'equivDiam': labeled_obj['equivalent_diameter_area'],
                    'eccentricity': labeled_obj['eccentricity'],
                    'circularity': circularity(labeled_obj),
                    'solidity': labeled_obj['solidity'],
                    'extent': labeled_obj['extent'],
                    'majorAxisLength': labeled_obj['major_axis_length'],
                    'minorAxisLength': labeled_obj['minor_axis_length'],
                    'minmajAxisRatio': minor_major_axis_ratio(labeled_obj),
                    'aspectRatio': aspect_ratio(labeled_obj),
                    'concavity': convex_hull_area_resid(labeled_obj),
                    'concavity_count': concavity_count(labeled_obj, min_area_fraction=0.005),
                    'asymmetry': convex_hull_centroid_dif(labeled_obj),
                    'disconnected': disconnected,
                }

            elif ot == 'organoid':
                # include only channel features
                row = {
                    'x_pos_weighted_pix': labeled_obj['weighted_centroid'][1],
                    'y_pos_weighted_pix': labeled_obj['weighted_centroid'][0],
                    'x_massDisp_pix': labeled_obj['weighted_centroid'][1] - labeled_obj['centroid'][1],
                    'y_massDisp_pix': labeled_obj['weighted_centroid'][0] - labeled_obj['centroid'][0],
                    'abs_min': abs_min_intensity,
                }

            elif ot == 'nucleus' and channel == nuc_channel:
                # include both channel and shape features
                row = {
                    'nuc_id': int(labeled_obj['label']),
                    'segmentation_nuc': organoid.segmentations[nuc_ending],
                    'x_pos_vox': labeled_obj['centroid'][2],
                    'y_pos_vox': labeled_obj['centroid'][1],
                    'z_pos_vox': labeled_obj['centroid'][0],
                    'volume_pix': labeled_obj['area'],
                    'surface_area': labeled_obj['surface_area_marchingcube'],
                    'equivDiam': labeled_obj['equivalent_diameter_area'],
                    'solidity': labeled_obj['solidity'],
                    'extent': labeled_obj['extent'],
                    'majorAxisLength': labeled_obj['major_axis_length'],
                    'minorAxisLength': labeled_obj['minor_axis_length'],
                    'minmajAxisRatio': minor_major_axis_ratio(labeled_obj),
                    'aspectRatio': aspect_ratio(labeled_obj),
                }

            elif ot == 'nucleus':
                # include only channel features
                row = {
                    'nuc_id': int(labeled_obj['label']),
                    'segmentation_nuc': organoid.segmentations[nuc_ending],
                }

            elif ot == 'membrane' and channel == mem_channel:
                # include both channel and shape features
                row = {
                    'mem_id': int(labeled_obj['label']),
                    'segmentation_mem': organoid.segmentations[mem_ending],
                    'x_pos_vox': labeled_obj['centroid'][2],
                    'y_pos_vox': labeled_obj['centroid'][1],
                    'z_pos_vox': labeled_obj['centroid'][0],
                    'volume_pix': labeled_obj['area'],
                    'surface_area': labeled_obj['surface_area_marchingcube'],
                    'equivDiam': labeled_obj['equivalent_diameter_area'],
                    'solidity': labeled_obj['solidity'],
                    'extent': labeled_obj['extent'],
                    'majorAxisLength': labeled_obj['major_axis_length'],
                    'minorAxisLength': labeled_obj['minor_axis_length'],
                    'minmajAxisRatio': minor_major_axis_ratio(labeled_obj),
                    'aspectRatio': aspect_ratio(labeled_obj),
                }

            elif ot == 'membrane':
                # include only channel features
                row = {
                    'mem_id': int(labeled_obj['label']),
                    'segmentation_mem': organoid.segmentations[mem_ending],
                }
            else:
                raise RuntimeError('unrecognized object type')

        except Exception as e:
            print_exc()

        common_row.update(row)
        row_list.append(common_row)

    df = pd.DataFrame(row_list)

    return df


def regionprops_to_row_organoid(**kwargs):
    return regionprops_to_row('organoid', **kwargs)


def regionprops_to_row_nucleus(**kwargs):
    return regionprops_to_row('nucleus', **kwargs)


def regionprops_to_row_membrane(**kwargs):
    return regionprops_to_row('membrane', **kwargs)



