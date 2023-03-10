# Copyright (C) 2023 Friedrich Miescher Institute for Biomedical Research

##############################################################################
#                                                                            #
# Author: Nicole Repina              <nicole.repina@fmi.ch>                  #
# Author: Tim-Oliver Buchholz        <tim-oliver.buchholz@fmi.ch>            #
# Author: Enrico Tagliavini          <enrico.tagliavini@fmi.ch>              #
#                                                                            #
##############################################################################

import copy
import logging
import os
from os.path import basename, join
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from faim_hcs.hcs.Experiment import Experiment
from faim_hcs.records.WellRecord import WellRecord
from scipy.ndimage import shift
from skimage.io import imsave
from skimage.registration import phase_cross_correlation

from scmultiplex.linking.matching import matching


def link_organoids(
    well: WellRecord,
    ovr_channel: str,
    folder_name: str,
    R0: Experiment,
    RX: Experiment,
    seg_name: str,
    RX_name: str,
    logger=logging,
):
    well_id = well.well_id
    plate_id = well.plate.plate_id

    #     if well.plate.plate_id not in ["day2"]:
    #         continue #skip these timepoints

    R0_fname = basename(well.segmentations[ovr_channel])
    R0_savedir = os.path.join(
        R0.get_experiment_dir(),
        well.plate.plate_id,
        well.well_id,
        "TIF_OVR_MIP_SEG",
        folder_name,
    )

    RX_fname = basename(RX.plates[plate_id].wells[well_id].segmentations[ovr_channel])
    RX_savedir = os.path.join(
        RX.get_experiment_dir(),
        well.plate.plate_id,
        well.well_id,
        "TIF_OVR_MIP_SEG",
        folder_name,
    )

    # load overviews
    R0_ovr = well.get_segmentation(ovr_channel)[0, :, :]
    RX_ovr = RX.plates[plate_id].wells[well_id].get_segmentation(ovr_channel)[0, :, :]

    # pad to make same shape
    R0_pad = np.pad(
        R0_ovr,
        [
            (0, max(0, RX_ovr.shape[0] - R0_ovr.shape[0])),
            (0, max(0, RX_ovr.shape[1] - R0_ovr.shape[1])),
        ],
        mode="constant",
        constant_values=0,
    )
    RX_pad = np.pad(
        RX_ovr,
        [
            (0, max(0, R0_ovr.shape[0] - RX_ovr.shape[0])),
            (0, max(0, R0_ovr.shape[1] - RX_ovr.shape[1])),
        ],
        mode="constant",
        constant_values=0,
    )

    if (R0_pad.shape[0] != RX_pad.shape[0]) | (R0_pad.shape[1] != RX_pad.shape[1]):
        logger.error("Check zero-padding")

    # binarize padded overviews
    R0_pad_binary = copy.deepcopy(R0_pad)
    R0_pad_binary[R0_pad_binary > 0] = 1  # binarize image

    RX_pad_binary = copy.deepcopy(RX_pad)
    RX_pad_binary[RX_pad_binary > 0] = 1  # binarize image

    # bin so that registration runs faster
    R0_pad_binary_bin = R0_pad_binary[::4, ::4]
    RX_pad_binary_bin = RX_pad_binary[::4, ::4]

    # calculate shifts
    # returns shift vector (in pixels) required to register moving_image (RX) with reference_image (R0)
    shifts = phase_cross_correlation(R0_pad_binary_bin, RX_pad_binary_bin)  # (y,x)

    # apply correction to RX overview
    RX_pad_binary_bin_shifted = shift(
        RX_pad_binary_bin, shifts[0], mode="constant", cval=0
    )
    RX_pad_shifted = shift(
        RX_pad, 4 * shifts[0], mode="constant", cval=0
    )  # use only for visualization to check alignment

    # save
    if not os.path.exists(R0_savedir):
        os.makedirs(R0_savedir)
    if not os.path.exists(RX_savedir):
        os.makedirs(RX_savedir)

    # plot
    logger.info(f"{plate_id}, {well_id}, shifts: {4 * shifts[0]}")
    plt.imsave(
        join(
            RX_savedir, f"Plate_{plate_id}_{well_id}_R0_" f"" f"{RX_name}_corrected.png"
        ),
        R0_pad_binary_bin + RX_pad_binary_bin_shifted,
    )

    R0_seg_file = os.path.join(R0_savedir, R0_fname)
    imsave(
        R0_seg_file,
        R0_pad.astype(np.int16),
        check_contrast=False,
    )
    RX_seg_file = os.path.join(RX_savedir, RX_fname)
    imsave(
        RX_seg_file,
        RX_pad_shifted.astype(np.int16),
        check_contrast=False,
    )

    # add to hcs experiments (R0 and RX)
    # R0_seg_file = os.path.join(
    #     well.plate.plate_id, "TIF_OVR_MIP_SEG", folder_name, R0_fname
    # )
    # RX_seg_file = os.path.join(
    #     well.plate.plate_id, "TIF_OVR_MIP_SEG", folder_name, RX_fname
    # )

    #     well.add_segmentation(seg_name, R0_seg_file) #add to R0
    #     RX.plates[plate_id].wells[well_id].add_segmentation(seg_name, RX_seg_file) #add to RX

    try:
        # Add the measurement to the faim-hcs datastructure
        well.add_segmentation(seg_name, R0_seg_file)  # add to R0
        well.save()  # updates json file

    except Exception as e:
        logger.error(e)

    try:
        # Add the measurement to the faim-hcs datastructure
        RX.plates[plate_id].wells[well_id].add_segmentation(
            seg_name, RX_seg_file
        )  # add to RX
        RX.plates[plate_id].wells[well_id].save()  # updates json file

    except Exception as e:
        logger.error(e)


def get_linking_stats(
    well: WellRecord,
    seg_name: str,
    RX: Experiment,
    iou_cutoff: float,
    names: List[str],
    ovr_channel: str,
    logger=logging,
):

    plate_id = well.plate.plate_id
    well_id = well.well_id

    logger.info(f"{plate_id}, {well_id}")

    # R0_ovr_seg = well.get_segmentation(ovr_channel)[0,:,:] #uncomment this is running withour registration
    R0_ovr_seg = well.get_segmentation(seg_name)  # load the seg image

    # load RX data
    try:
        RX_ovr_seg = RX.plates[plate_id].wells[well_id].get_segmentation(seg_name)
    except Exception as e:
        logger.error(e)

    stat = matching(
        R0_ovr_seg, RX_ovr_seg, criterion="iou", thresh=iou_cutoff, report_matches=True
    )
    # print(stat)
    logger.info(f"{stat[2]} out of {stat[10]} RX_org are not matched to an " "R0_org.")
    logger.info(f"{stat[4]} out of {stat[9]} R0_org are not matched to an " "RX_org.")

    df = pd.DataFrame(
        list(zip([x[0] for x in stat[14]], [x[1] for x in stat[14]], stat[15])),
        columns=["R0_label", "RX_label", "iou"],
    )
    df_filt = df[df["iou"] > iou_cutoff]
    logger.info(
        f"removed {len(df) - len(df_filt)} our of {len(df)} RX "
        f"organoids that are not matched to R0."
    )

    df_filt = df_filt.sort_values(by=["R0_label"])

    df_filt["hcs_experiment_" + names[0]] = well.plate.experiment.name
    df_filt["root_dir_" + names[0]] = well.plate.experiment.root_dir
    df_filt["segmentation_ovr_" + names[0]] = well.segmentations[ovr_channel]
    df_filt["well_id"] = well_id
    df_filt["plate_id"] = plate_id
    df_filt["channel_id"] = ovr_channel

    df_filt["hcs_experiment_" + names[1]] = RX.name
    df_filt["root_dir_" + names[1]] = RX.root_dir
    df_filt["segmentation_ovr_" + names[1]] = (
        RX.plates[plate_id].wells[well_id].segmentations[ovr_channel]
    )

    # df_save measurement into the well directory.
    name = "linking_ovr_" + str(ovr_channel) + "_" + names[1] + "to" + names[0]
    path = join(well.well_dir, name + ".csv")
    df_filt.to_csv(path, index=False)  # df_saves csv

    try:
        # Add the measurement to the faim-hcs datastructure
        well.add_measurement(name, path)
        well.save()  # updates json file

    except Exception as e:
        logger.error(e)
