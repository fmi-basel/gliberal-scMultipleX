import copy
import math
from os.path import join
from typing import List

import numpy as np
import pandas as pd
from faim_hcs.hcs.Experiment import Experiment
from faim_hcs.records.OrganoidRecord import OrganoidRecord
from faim_hcs.records.WellRecord import WellRecord
from skimage.measure import regionprops

from scmultiplex.features.FeatureFunctions import (
    fixed_percentiles,
    kurtos,
    skewness,
    stdv,
)
from scmultiplex.linking.matching import matching


def extract_2d_ovr(
    well: WellRecord, ovr_channel: str, ovr_seg_img, touching_labels: List[str]
):
    df_ovr = pd.DataFrame()
    ovr_features = regionprops(ovr_seg_img)

    for obj in ovr_features:
        organoid_id = "object_" + str(obj["label"])
        row = {
            "hcs_experiment": well.plate.experiment.name,
            "plate_id": well.plate.plate_id,
            "well_id": well.well_id,
            "organoid_id": organoid_id,
            "segmentation_ovr": well.segmentations[ovr_channel],
            "flag_tile_border": organoid_id
            in touching_labels,  # TRUE (1) means organoid is touching a tile border
            "x_pos_pix_global": obj["centroid"][1],
            "y_pos_pix_global": obj["centroid"][0],
            "area_pix_global": obj["area"],
        }

        df_ovr = pd.concat(
            [df_ovr, pd.DataFrame.from_records([row])], ignore_index=True
        )
    return df_ovr


def extract_organoid_features(
    organoid: OrganoidRecord, nuc_ending: str, mem_ending: str, mask_ending: str
):
    nuc_seg = organoid.get_segmentation(nuc_ending)  # load segmentation images
    mem_seg = organoid.get_segmentation(mem_ending)  # load segmentation images
    org_seg = organoid.get_segmentation(mask_ending)

    # for each raw image, extract organoid-level features
    for channel in organoid.raw_files:
        # Load raw data.
        raw = organoid.get_raw_data(channel)  # this is the raw image

        # create organoid MIP
        raw_mip = np.zeros(
            org_seg.shape
        )  # initialize array to dimensions of mask image
        for plane in raw:
            raw_mip = np.maximum(raw_mip, plane)

        # organoid feature extraction
        df_org = pd.DataFrame()
        org_features = regionprops(
            org_seg,
            raw_mip,
            extra_properties=(fixed_percentiles, skewness, kurtos, stdv),
        )
        abs_min_intensity = np.amin(raw_mip)
        # voxel_area = organoid.spacings[channel][1] * organoid.spacings[channel][2] #calculate voxel area in um2 (x*y)

        for obj in org_features:
            box_area = obj["area"] / (
                org_seg.shape[0] * org_seg.shape[1]
            )  # for filtering of wrong segmentations
            circularity = 4 * math.pi * (obj["area"] / (math.pow(obj["perimeter"], 2)))

            row = {
                "hcs_experiment": organoid.well.plate.experiment.name,
                "root_dir": organoid.well.plate.experiment.root_dir,
                "plate_id": organoid.well.plate.plate_id,
                "well_id": organoid.well.well_id,
                "channel_id": channel,
                "object_type": "organoid",
                "organoid_id": organoid.organoid_id,
                "org_label": organoid.organoid_id.rpartition("_")[2],
                "segmentation_org": organoid.segmentations[mask_ending],
                "intensity_img": organoid.raw_files[channel],
                "x_pos_pix": obj["centroid"][1],
                "y_pos_pix": obj["centroid"][0],
                "x_pos_weighted_pix": obj["weighted_centroid"][1],
                "y_pos_weighted_pix": obj["weighted_centroid"][0],
                "x_massDisp_pix": obj["weighted_centroid"][1] - obj["centroid"][1],
                "y_massDisp_pix": obj["weighted_centroid"][0] - obj["centroid"][0],
                "mean_intensityMIP": obj["mean_intensity"],
                "max_intensity": obj["max_intensity"],
                "min_intensity": obj["min_intensity"],
                "abs_min": abs_min_intensity,
                "area_pix": obj["area"],
                #                 'area_um2':obj['area'] * voxel_area,
                "eccentricity": obj["eccentricity"],
                "majorAxisLength": obj["major_axis_length"],
                "minorAxisLength": obj["minor_axis_length"],
                "axisRatio": obj["minor_axis_length"] / obj["major_axis_length"],
                "eulerNumber": obj["euler_number"],
                # for filtering wrong segmentations
                "objectBoxRatio": box_area,
                # for filtering wrong segmentations
                "perimeter": obj["perimeter"],
                "circularity": circularity,
                "quartile25": obj["fixed_percentiles"][0],
                "quartile50": obj["fixed_percentiles"][1],
                "quartile75": obj["fixed_percentiles"][2],
                "quartile90": obj["fixed_percentiles"][3],
                "quartile95": obj["fixed_percentiles"][4],
                "quartile99": obj["fixed_percentiles"][5],
                "stdev": obj["stdv"],
                "skew": obj["skewness"],
                "kurtosis": obj["kurtos"],
            }

            df_org = pd.concat(
                [df_org, pd.DataFrame.from_records([row])], ignore_index=True
            )
            # df_org = df_org.append(row,ignore_index=True)

        # Save measurement into the organoid directory.
        name = "regionprops_org_" + str(channel)
        path = join(organoid.organoid_dir, name + ".csv")
        df_org.to_csv(path, index=False)  # saves csv

        # Add the measurement to the faim-hcs datastructure and save.
        organoid.add_measurement(name, path)
        organoid.save()  # updates json file

        # NUCLEAR feature extraction
        if nuc_seg is None:
            continue  # skip organoids that don't have a nuclear segmentation

        # organoid feature extraction
        df_nuc = pd.DataFrame()

        # make binary organoid mask and crop nuclear labels to this mask
        # extract nuclei features that only belong to organoid of interest and exclude pieces of neighboring organoids
        org_seg_binary = copy.deepcopy(org_seg)
        org_seg_binary[org_seg_binary > 0] = 1
        nuc_seg = nuc_seg * org_seg_binary

        nuc_features = regionprops(
            nuc_seg,
            raw,
            extra_properties=(fixed_percentiles, skewness, kurtos, stdv),
            spacing=(3, 1, 1),
        )
        # voxel_volume = organoid.spacings[channel][0] * organoid.spacings[channel][1] * organoid.spacings[channel][2] #calculate voxel area in um2 (x*y)
        # https://www.analyticsvidhya.com/blog/2022/01/moments-a-must-known-statistical-concept-for-data-science/
        # mean is first raw moment
        # variance is the second central moment
        # skewness is the third normalized moment
        # kurtosis is the fourth standardize moment

        for nuc in nuc_features:
            row = {
                "hcs_experiment": organoid.well.plate.experiment.name,
                "root_dir": organoid.well.plate.experiment.root_dir,
                "plate_id": organoid.well.plate.plate_id,
                "well_id": organoid.well.well_id,
                "channel_id": channel,
                "object_type": "nucleus",
                "organoid_id": organoid.organoid_id,
                "org_label": organoid.organoid_id.rpartition("_")[2],
                "nuc_id": int(nuc["label"]),
                "segmentation_nuc": organoid.segmentations[nuc_ending],
                "intensity_img": organoid.raw_files[channel],
                "x_pos_vox": nuc["centroid"][2],
                "y_pos_vox": nuc["centroid"][1],
                "z_pos_vox": nuc["centroid"][0],
                #                 'x_pos_um':nuc['centroid'][2]*organoid.spacings[channel][2],
                #                 'y_pos_um':nuc['centroid'][1]*organoid.spacings[channel][1],
                #                 'z_pos_um':nuc['centroid'][0]*organoid.spacings[channel][0],
                "volume_pix": nuc["area"],
                #                 'volume_um3':nuc['area'] * voxel_volume,
                "mean_intensity": nuc["mean_intensity"],
                "max_intensity": nuc["max_intensity"],
                "min_intensity": nuc["min_intensity"],
                "quartile25": nuc["fixed_percentiles"][0],
                "quartile50": nuc["fixed_percentiles"][1],
                "quartile75": nuc["fixed_percentiles"][2],
                "quartile90": nuc["fixed_percentiles"][3],
                "quartile95": nuc["fixed_percentiles"][4],
                "quartile99": nuc["fixed_percentiles"][5],
                "stdev": nuc["stdv"],
                "skew": nuc["skewness"],
                "kurtosis": nuc["kurtos"],
            }

            df_nuc = pd.concat(
                [df_nuc, pd.DataFrame.from_records([row])], ignore_index=True
            )
            # df_nuc = df_nuc.append(row,ignore_index=True)

        # Save measurement into the organoid directory.
        name = "regionprops_nuc_" + str(channel)
        path = join(organoid.organoid_dir, name + ".csv")
        df_nuc.to_csv(path, index=False)  # saves csv

        # Add the measurement to the faim-hcs datastructure and save.
        organoid.add_measurement(name, path)
        organoid.save()  # updates json file

        # MEMBRANE feature extraction
        if mem_seg is None:
            continue  # skip organoids that don't have a cell segmentation

        # organoid feature extraction
        df_mem = pd.DataFrame()

        # make binary organoid mask and crop cell labels to this mask
        # MAYBE EXPAND BINARY MASK??
        mem_seg = mem_seg * org_seg_binary

        mem_features = regionprops(
            mem_seg,
            raw,
            extra_properties=(fixed_percentiles, skewness, kurtos, stdv),
            spacing=(3, 1, 1),
        )

        for mem in mem_features:
            row = {
                "hcs_experiment": organoid.well.plate.experiment.name,
                "root_dir": organoid.well.plate.experiment.root_dir,
                "plate_id": organoid.well.plate.plate_id,
                "well_id": organoid.well.well_id,
                "channel_id": channel,
                "object_type": "membrane",
                "organoid_id": organoid.organoid_id,
                "org_label": organoid.organoid_id.rpartition("_")[2],
                "mem_id": int(mem["label"]),
                "segmentation_mem": organoid.segmentations[mem_ending],
                "intensity_img": organoid.raw_files[channel],
                "x_pos_vox": mem["centroid"][2],
                "y_pos_vox": mem["centroid"][1],
                "z_pos_vox": mem["centroid"][0],
                #                 'x_pos_um':mem['centroid'][2]*organoid.spacings[channel][2],
                #                 'y_pos_um':mem['centroid'][1]*organoid.spacings[channel][1],
                #                 'z_pos_um':mem['centroid'][0]*organoid.spacings[channel][0],
                "volume_pix": mem["area"],
                #                 'volume_um3':mem['area'] * voxel_volume,
                "mean_intensity": mem["mean_intensity"],
                "max_intensity": mem["max_intensity"],
                "min_intensity": mem["min_intensity"],
                "quartile25": mem["fixed_percentiles"][0],
                "quartile50": mem["fixed_percentiles"][1],
                "quartile75": mem["fixed_percentiles"][2],
                "quartile90": mem["fixed_percentiles"][3],
                "quartile95": mem["fixed_percentiles"][4],
                "quartile99": mem["fixed_percentiles"][5],
                "stdev": mem["stdv"],
                "skew": mem["skewness"],
                "kurtosis": mem["kurtos"],
            }

            df_mem = pd.concat(
                [df_mem, pd.DataFrame.from_records([row])], ignore_index=True
            )
            # df_mem = df_mem.append(row,ignore_index=True)

        # Save measurement into the organoid directory.
        name = "regionprops_mem_" + str(channel)
        path = join(organoid.organoid_dir, name + ".csv")
        df_mem.to_csv(path, index=False)  # saves csv

        # Add the measurement to the faim-hcs datastructure and save.
        organoid.add_measurement(name, path)
        organoid.save()  # updates json file


def link_nuc_to_membrane(
    exp: Experiment,
    ovr_channel: str,
    nuc_ending: str,
    mask_ending: str,
    mem_ending: str,
    iop_cutoff: float,
):
    exp.only_iterate_over_wells(False)
    exp.reset_iterator()

    for organoid in exp:
        # nuclear feature extraction
        nuc_seg = organoid.get_segmentation(nuc_ending)  # load segmentation images
        mem_seg = organoid.get_segmentation(mem_ending)  # load segmentation images
        # org_seg = organoid.get_segmentation("MASK")
        org_seg = organoid.get_segmentation(mask_ending)

        if nuc_seg is None:
            continue  # skip organoids that don't have a nuclear segmentation
        if mem_seg is None:
            continue  # skip organoids that don't have a membrane segmentation

        org_seg_binary = copy.deepcopy(org_seg)
        org_seg_binary[org_seg_binary > 0] = 1

        nuc_seg = nuc_seg * org_seg_binary
        mem_seg = mem_seg * org_seg_binary

        # match each nuclear label to a cell label
        stat = matching(
            mem_seg, nuc_seg, criterion="iop", thresh=iop_cutoff, report_matches=True
        )

        #     print(stat[2], 'out of', stat[10], 'nuclei are not surrounded by a cell')
        #     print(stat[4], 'out of', stat[9], 'cells do not contain a nucleus')

        match = pd.DataFrame(
            list(zip([x[0] for x in stat[14]], [x[1] for x in stat[14]], stat[15])),
            columns=["mem_id", "nuc_id", "iop"],
        )
        match_filt = match.loc[(match["iop"] > iop_cutoff)].copy(
            deep=True
        )  # this is the linking df

        # update all organoid measurements with numbers of nuclei and membrane detected and linked
        for meas_name in [
            k
            for k, v in organoid.measurements.items()
            if k.startswith("regionprops_org")
        ]:
            meas = organoid.get_measurement(meas_name)

            # add columns to dataframe
            meas["nuc_without_mem"] = stat[2]
            meas["nuc_total"] = stat[10]
            meas["mem_without_nuc"] = stat[4]
            meas["mem_total"] = stat[9]

            name = str(meas_name)
            path = join(organoid.organoid_dir, name + ".csv")
            meas.to_csv(path, index=False)  # saves csv

            # Add the measurement to the faim-hcs datastructure and save.
            organoid.add_measurement(name, path)
            organoid.save()  # updates json file

        # add metadata and save linking
        match_filt["hcs_experiment"] = organoid.well.plate.experiment.name
        match_filt["root_dir"] = organoid.well.plate.experiment.root_dir
        match_filt["plate_id"] = organoid.well.plate.plate_id
        match_filt["well_id"] = organoid.well.well_id
        match_filt["channel_id"] = ovr_channel
        match_filt["organoid_id"] = organoid.organoid_id
        match_filt["org_label"] = organoid.organoid_id.rpartition("_")[2]

        # Save measurement into the organoid directory.
        name = "linking_nuc_to_mem"
        path = join(organoid.organoid_dir, name + ".csv")
        match_filt.to_csv(path, index=False)  # saves csv

        # Add the measurement to the faim-hcs datastructure and save.
        organoid.add_measurement(name, path)
        organoid.save()  # updates json file
