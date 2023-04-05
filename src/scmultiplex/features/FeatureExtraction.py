# Copyright (C) 2023 Friedrich Miescher Institute for Biomedical Research

##############################################################################
#                                                                            #
# Author: Nicole Repina              <nicole.repina@fmi.ch>                  #
# Author: Tim-Oliver Buchholz        <tim-oliver.buchholz@fmi.ch>            #
# Author: Enrico Tagliavini          <enrico.tagliavini@fmi.ch>              #
#                                                                            #
##############################################################################

import copy
from os.path import join
from typing import List, Tuple

import numpy as np
import pandas as pd
from scmultiplex.faim_hcs.records.OrganoidRecord import OrganoidRecord
from scmultiplex.faim_hcs.records.WellRecord import WellRecord

from scmultiplex.features.FeatureFunctions import flag_touching

from scmultiplex.features.FeatureProps import measure_features

from scmultiplex.linking.matching import matching

from scmultiplex.utils.load_utils import load_ovr


def extract_organoid_features(
    organoid: OrganoidRecord,
    nuc_ending: str,
    mem_ending: str,
    mask_ending: str,
    spacing: Tuple[float],
    measure_morphology,
    organoid_seg_channel,
    nuclear_seg_channel,
    membrane_seg_channel,
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

        abs_min_intensity = np.amin(raw_mip)

        calc_morphology = (channel == organoid_seg_channel) and measure_morphology

        extra_values_common = {'hcs_experiment': organoid.well.plate.experiment.name,
                               'root_dir': organoid.well.plate.experiment.root_dir,
                               'plate_id': organoid.well.plate.plate_id,
                               'well_id': organoid.well.well_id,
                               'channel_id': channel,
                               'org_id': int(organoid.organoid_id.rpartition("_")[2]),
                               'intensity_img': organoid.raw_files[channel],
                               }
        extra_values_org = {
            'segmentation_org': organoid.segmentations[mask_ending],
            'object_type': 'organoid',
            'abs_min': abs_min_intensity,
        }


        measure_features(
            object_type = 'org',
            record = organoid,
            channel = channel,
            label_img = org_seg,
            img = raw_mip,
            spacing = spacing,
            is_2D = True,
            measure_morphology = calc_morphology,
            min_area_fraction = 0.005,
            channel_prefix = None,
            extra_values_common = extra_values_common,
            extra_values_object = extra_values_org,
            touching_labels = None,
            )

        # NUCLEAR feature extraction
        if nuc_seg is not None:
            calc_morphology = (channel == nuclear_seg_channel) and measure_morphology

            extra_values_nuc = {
                'segmentation_nuc': organoid.segmentations[nuc_ending],
                'object_type': 'nucleus'
            }

            # make binary organoid mask and crop nuclear labels to this mask to limit nuclei from neighboring orgs
            org_seg_binary = copy.deepcopy(org_seg)
            org_seg_binary[org_seg_binary > 0] = 1
            nuc_seg = nuc_seg * org_seg_binary

            measure_features(
                object_type='nuc',
                record=organoid,
                channel=channel,
                label_img=nuc_seg,
                img=raw,
                spacing=spacing,
                is_2D=False,
                measure_morphology=calc_morphology,
                min_area_fraction=0.005,
                channel_prefix=None,
                extra_values_common=extra_values_common,
                extra_values_object=extra_values_nuc,
                touching_labels=None,
            )

        # MEMBRANE feature extraction
        if mem_seg is not None:
            calc_morphology = (channel == membrane_seg_channel) and measure_morphology

            extra_values_mem = {
                'segmentation_mem': organoid.segmentations[mem_ending],
                'object_type': 'membrane'
            }

            org_seg_binary = copy.deepcopy(org_seg)
            org_seg_binary[org_seg_binary > 0] = 1
            mem_seg = mem_seg * org_seg_binary

            measure_features(
                object_type='mem',
                record=organoid,
                channel=channel,
                label_img=mem_seg,
                img=raw,
                spacing=spacing,
                is_2D=False,
                measure_morphology=calc_morphology,
                min_area_fraction=0.005,
                channel_prefix=None,
                extra_values_common=extra_values_common,
                extra_values_object=extra_values_mem,
                touching_labels=None,
            )

    return


def extract_well_features(
    well: WellRecord,
    ovr_channel: str,
):
    ovr_seg_img, ovr_seg_tiles = load_ovr(well, ovr_channel)

    touching_labels = flag_touching(ovr_seg_img, ovr_seg_tiles)

    # if ovr_seg_img is None and ovr_seg_tiles is None:
    #     logger.warning(f"ovr_seg does not exists. Skipping {well.well_id}.")

    extra_values_common = {'hcs_experiment': well.plate.experiment.name,
                           'plate_id': well.plate.plate_id,
                           'well_id': well.well_id,
                           }

    extra_values_ovr = {
        "segmentation_ovr": well.segmentations[ovr_channel],
    }

    measure_features(
        object_type='ovr',
        record=well,
        channel=ovr_channel,
        label_img=ovr_seg_img,
        img=None,
        spacing=None,
        is_2D=True,
        measure_morphology=False,
        min_area_fraction=0.005,
        channel_prefix=None,
        extra_values_common=extra_values_common,
        extra_values_object=extra_values_ovr,
        touching_labels=touching_labels,
    )

    return


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
