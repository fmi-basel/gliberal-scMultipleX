import copy
import os
from os.path import basename, join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from faim_hcs.hcs.Experiment import Experiment
from scipy.ndimage import shift
from skimage.io import imsave
from skimage.registration import phase_cross_correlation

RX_name = "R2"
# must be relative to R0
R0_dir = "/tungstenfs/scratch/gliberal/Users/repinico/Microscopy/Analysis/20220528_GCPLEX_redo/20220507GCPLEX_R0/summary.csv"
RX_dir = "/tungstenfs/scratch/gliberal/Users/repinico/Microscopy/Analysis/20220528_GCPLEX_redo/20220507GCPLEX_R2/summary.csv"

iou_cutoff = 0.2

ovr_channel = "C01"  # almost always DAPI is C01 -- if not, change this!

# load experiments
R0 = Experiment()
R0.load(R0_dir)

RX = Experiment()
RX.load(RX_dir)

names = ["R0", RX_name]
exps = [R0, RX]

zipd = zip(names, exps)
rounds = dict(zipd)

seg_name = RX_name + "_linked"
folder_name = "obj_v0.3_registered_" + RX_name

## Pre-register OVR images, save, and add to hcs experiment
R0.only_iterate_over_wells(True)
R0.reset_iterator()

for well in R0:
    well_id = well.well_id
    plate_id = well.plate.plate_id

    #     if well.plate.plate_id not in ["day2"]:
    #         continue #skip these timepoints

    R0_fname = basename(well.segmentations[ovr_channel])
    R0_savedir = os.path.join(
        well.plate.experiment.root_dir,
        well.plate.plate_id,
        "TIF_OVR_MIP_SEG",
        folder_name,
    )

    RX_fname = basename(RX.plates[plate_id].wells[well_id].segmentations[ovr_channel])
    RX_savedir = os.path.join(
        RX.root_dir, well.plate.plate_id, "TIF_OVR_MIP_SEG", folder_name
    )

    #     print(R0_fname, RX_fname)
    #     print(R0_savedir,RX_savedir)

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
        print("error! check zero-padding")

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

    # plot
    print(plate_id, well_id, "shifts:", 4 * shifts[0])
    plt.figure(figsize=(10, 10))
    plt.imshow(R0_pad_binary_bin + RX_pad_binary_bin_shifted)
    plt.title("R0 + RX corrected, binary and 4x binned")
    plt.show()

    # save
    if not os.path.exists(R0_savedir):
        os.makedirs(R0_savedir)
    if not os.path.exists(RX_savedir):
        os.makedirs(RX_savedir)

    imsave(
        os.path.join(R0_savedir, R0_fname),
        R0_pad.astype(np.int16),
        check_contrast=False,
    )
    imsave(
        os.path.join(RX_savedir, RX_fname),
        RX_pad_shifted.astype(np.int16),
        check_contrast=False,
    )

    # add to hcs experiments (R0 and RX)
    R0_seg_file = os.path.join(
        well.plate.plate_id, "TIF_OVR_MIP_SEG", folder_name, R0_fname
    )
    RX_seg_file = os.path.join(
        well.plate.plate_id, "TIF_OVR_MIP_SEG", folder_name, RX_fname
    )

    #     well.add_segmentation(seg_name, R0_seg_file) #add to R0
    #     RX.plates[plate_id].wells[well_id].add_segmentation(seg_name, RX_seg_file) #add to RX

    try:
        # Add the measurement to the faim-hcs datastructure
        well.add_segmentation(seg_name, R0_seg_file)  # add to R0
        well.save()  # updates json file

    except Exception as e:
        print(e)

    try:
        # Add the measurement to the faim-hcs datastructure
        RX.plates[plate_id].wells[well_id].add_segmentation(
            seg_name, RX_seg_file
        )  # add to RX
        RX.plates[plate_id].wells[well_id].save()  # updates json file

    except Exception as e:
        print(e)


# perform linking


R0.only_iterate_over_wells(True)
R0.reset_iterator()

for well in R0:  # for each well that is in R0

    #     if well.plate.plate_id not in ["day4"]:
    #         #print("skipping", organoid.well.plate.plate_id)
    #         continue #skip these timepoints

    df_filt = None

    plate_id = well.plate.plate_id
    well_id = well.well_id

    print(plate_id, well_id)

    # load R0 data

    # R0_ovr_seg = well.get_segmentation(ovr_channel)[0,:,:] #uncomment this is running withour registration
    R0_ovr_seg = well.get_segmentation(seg_name)  # load the seg image

    # load RX data
    try:
        RX_ovr_seg = RX.plates[plate_id].wells[well_id].get_segmentation(seg_name)
    except Exception as e:
        print(e)
        continue

    stat = matching(
        R0_ovr_seg, RX_ovr_seg, criterion="iou", thresh=iou_cutoff, report_matches=True
    )
    # print(stat)
    print(stat[2], "out of", stat[10], "RX_org are not matched to an R0_org")
    print(stat[4], "out of", stat[9], "R0_org are not matched to an RX_org")

    df = pd.DataFrame(
        list(zip([x[0] for x in stat[14]], [x[1] for x in stat[14]], stat[15])),
        columns=["R0_label", "RX_label", "iou"],
    )
    df_filt = df[df["iou"] > iou_cutoff]
    print(
        "removed",
        len(df) - len(df_filt),
        "out of",
        len(df),
        "RX organoids that are not matched to R0",
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

    display(df_filt)
    # df_save measurement into the well directory.
    name = "linking_ovr_NEW_" + str(ovr_channel) + "_" + names[1] + "to" + names[0]
    path = join(well.well_dir, name + ".csv")
    df_filt.to_csv(path, index=False)  # df_saves csv

    try:
        # Add the measurement to the faim-hcs datastructure
        well.add_measurement(name, path)
        well.save()  # updates json file

    except Exception as e:
        print(e)

print("Saved!")
