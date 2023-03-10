import copy
from os.path import join
from typing import List, Tuple

import numpy as np
import pandas as pd
from faim_hcs.records.OrganoidRecord import OrganoidRecord
from faim_hcs.records.WellRecord import WellRecord
from skimage.measure import regionprops

from scmultiplex.features.FeatureFunctions import (
    fixed_percentiles,
    kurtos,
    skewness,
    stdv,
    disconnected_component,
    surface_area_marchingcube,
)
from scmultiplex.features.FeatureProps import (
    regionprops_to_row_organoid,
    regionprops_to_row_nucleus,
    regionprops_to_row_membrane
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
            "org_id": int(organoid_id.rpartition("_")[2]),
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
    organoid: OrganoidRecord,
    nuc_ending: str,
    mem_ending: str,
    mask_ending: str,
    spacing: Tuple[float],
    org_seg_ch,
    nuc_seg_ch,
    mem_seg_ch,
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
        org_features = regionprops(
            org_seg,
            raw_mip,
            extra_properties=(fixed_percentiles, skewness, kurtos, stdv),
        )

        abs_min_intensity = np.amin(raw_mip)
        img_dim = org_seg.shape
        disconnected = disconnected_component(org_seg)

        df_org = regionprops_to_row_organoid(regionproperties=org_features,
                                         org_channel=org_seg_ch,
                                         nuc_channel=nuc_seg_ch,
                                         mem_channel=mem_seg_ch,
                                         organoid=organoid,
                                         channel=channel,
                                         mask_ending=mask_ending,
                                         nuc_ending=nuc_ending,
                                         mem_ending=mem_ending,
                                         abs_min_intensity=abs_min_intensity,
                                         img_dim=img_dim,
                                         disconnected=disconnected)

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

        # make binary organoid mask and crop nuclear labels to this mask to limit nuclei from neighboting orgs
        org_seg_binary = copy.deepcopy(org_seg)
        org_seg_binary[org_seg_binary > 0] = 1
        nuc_seg = nuc_seg * org_seg_binary

        nuc_features = regionprops(
            nuc_seg,
            raw,
            extra_properties=(fixed_percentiles, skewness, kurtos, stdv, surface_area_marchingcube),
            spacing=spacing,
        )

        df_nuc = regionprops_to_row_nucleus(regionproperties=nuc_features,
                                         org_channel=org_seg_ch,
                                         nuc_channel=nuc_seg_ch,
                                         mem_channel=mem_seg_ch,
                                         organoid=organoid,
                                         channel=channel,
                                         mask_ending=mask_ending,
                                         nuc_ending=nuc_ending,
                                         mem_ending=mem_ending,
                                         abs_min_intensity=None,
                                         img_dim=None,
                                         disconnected=None)

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

        mem_seg = mem_seg * org_seg_binary

        mem_features = regionprops(
            mem_seg,
            raw,
            extra_properties=(fixed_percentiles, skewness, kurtos, stdv, surface_area_marchingcube),
            spacing=spacing,
        )

        df_mem = regionprops_to_row_membrane(regionproperties=mem_features,
                                         org_channel=org_seg_ch,
                                         nuc_channel=nuc_seg_ch,
                                         mem_channel=mem_seg_ch,
                                         organoid=organoid,
                                         channel=channel,
                                         mask_ending=mask_ending,
                                         nuc_ending=nuc_ending,
                                         mem_ending=mem_ending,
                                         abs_min_intensity=None,
                                         img_dim=None,
                                         disconnected=None)

        # Save measurement into the organoid directory.
        name = "regionprops_mem_" + str(channel)
        path = join(organoid.organoid_dir, name + ".csv")
        df_mem.to_csv(path, index=False)  # saves csv

        # Add the measurement to the faim-hcs datastructure and save.
        organoid.add_measurement(name, path)
        organoid.save()  # updates json file


def link_nuc_to_membrane(
    organoid: OrganoidRecord,
    ovr_channel: str,
    nuc_ending: str,
    mask_ending: str,
    mem_ending: str,
    iop_cutoff: float,
):

    nuc_seg = organoid.get_segmentation(nuc_ending)
    mem_seg = organoid.get_segmentation(mem_ending)
    org_seg = organoid.get_segmentation(mask_ending)

    if (nuc_seg is not None) and (mem_seg is not None):

        org_seg_binary = copy.deepcopy(org_seg)
        org_seg_binary[org_seg_binary > 0] = 1

        nuc_seg = nuc_seg * org_seg_binary
        mem_seg = mem_seg * org_seg_binary

        # match each nuclear label to a cell label
        stat = matching(
            mem_seg, nuc_seg, criterion="iop", thresh=iop_cutoff, report_matches=True
        )

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
        match_filt["org_id"] = organoid.organoid_id.rpartition("_")[2]

        # Save measurement into the organoid directory.
        name = "linking_nuc_to_mem"
        path = join(organoid.organoid_dir, name + ".csv")
        match_filt.to_csv(path, index=False)  # saves csv

        # Add the measurement to the faim-hcs datastructure and save.
        organoid.add_measurement(name, path)
        organoid.save()  # updates json file
