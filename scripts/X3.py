import warnings
from os.path import join

import pandas as pd
from faim_hcs.hcs.Experiment import Experiment
from run_platymatch_NAR220527 import runAffine, runFFD

pd.set_option("display.max_rows", None)


RX_name = "R1"
# must be relative to R0
R0_dir = "/tungstenfs/scratch/gliberal/Users/repinico/Microscopy/Analysis/20220528_GCPLEX_redo/20220507GCPLEX_R0/summary.csv"
RX_dir = "/tungstenfs/scratch/gliberal/Users/repinico/Microscopy/Analysis/20220528_GCPLEX_redo/20220507GCPLEX_R1/summary.csv"

segname = "NUC_SEG3D_220523"
ovr_channel = "C01"


R0 = Experiment()
R0.load(R0_dir)

RX = Experiment()
RX.load(RX_dir)

names = ["R0", RX_name]
exps = [R0, RX]

zipd = zip(names, exps)
rounds = dict(zipd)

R0.only_iterate_over_wells(False)
R0.reset_iterator()

for organoid in R0:
    affine_matches = None
    ffd_matches = None

    if organoid.well.plate.plate_id in ["day4p5"]:
        # print("skipping", organoid.well.plate.plate_id)
        continue  # skip these timepoints

    well_id = organoid.well.well_id
    plate_id = organoid.well.plate.plate_id

    # load R0 data
    R0_obj = organoid.organoid_id
    R0_id = int(R0_obj.rpartition("_")[2])

    #     if plate_id != "day3p5": #limit to 1 organoid
    #         continue

    #     if well_id != "B04": #limit to 1 organoid
    #         continue

    #     if R0_id != 4: #limit to 1 organoid
    #         continue

    # load organoid measurement data for filtering
    R0_df_ovr = organoid.well.get_measurement("regionprops_ovr_C01")
    R0_df_ovr = R0_df_ovr.set_index("organoid_id")
    R0_df_org = organoid.get_measurement("regionprops_org_C01")

    # load linking data
    link_org = organoid.well.get_measurement("linking_ovr_C01_" + RX_name + "toR0")
    link_org_dict = link_org.set_index("R0_label").T.to_dict("index")[
        "RX_label"
    ]  # R0 is key, RX is value

    # do not link organoids that don't have a RX-R0 match
    if R0_id not in link_org_dict:  # move to next organoid if no match
        continue

    # do not link border organoids
    if R0_df_ovr.loc[R0_obj, "flag_tile_border"] == True:
        # print("skipping", plate_id, well_id, R0_obj, "touches border")
        continue

    if R0_df_org["abs_min"][0] == 0:
        # print("skipping", plate_id, well_id, R0_obj, "min = 0")
        continue

    try:
        R0_df = organoid.get_measurement("regionprops_nuc_C01")
        R0_raw = organoid.get_raw_data(ovr_channel)
        R0_seg = organoid.get_segmentation(segname)
    except Exception as e:
        print(
            e
        )  # usually exception is that no nuc were detected so csv is empty. in this case, skip organoid
        continue

    # load RX data for organoids that pass
    RX_id = link_org_dict[R0_id]
    RX_obj = "object_" + str(RX_id)

    try:
        RX_df = (
            RX.plates[plate_id]
            .wells[well_id]
            .organoids[RX_obj]
            .get_measurement("regionprops_nuc_C01")
        )
        RX_raw = (
            RX.plates[plate_id]
            .wells[well_id]
            .organoids[RX_obj]
            .get_raw_data(ovr_channel)
        )
        RX_seg = (
            RX.plates[plate_id]
            .wells[well_id]
            .organoids[RX_obj]
            .get_segmentation(segname)
        )
    except Exception as e:
        print(
            e
        )  # usually exception is that no nuc were detected so csv is empty. in this case, skip organoid
        continue

        # convert df's to PlatyMatch compatible numpy array
    # N x 5 (first column is ids, last column is size)
    R0_numpy = R0_df[
        ["nuc_id", "x_pos_vox", "y_pos_vox", "z_pos_vox", "volume_pix"]
    ].to_numpy()
    RX_numpy = RX_df[
        ["nuc_id", "x_pos_vox", "y_pos_vox", "z_pos_vox", "volume_pix"]
    ].to_numpy()
    # Divid by z voxel anisotropy so that coordinates match label image!
    R0_numpy[:, 3] *= 1 / 3
    R0_numpy[:, 4] *= 1 / 3
    RX_numpy[:, 3] *= 1 / 3
    RX_numpy[:, 4] *= 1 / 3

    # skip organoids that have less than 4 nuclei
    if (R0_numpy.shape[0] <= 4) | (
        RX_numpy.shape[0] <= 4
    ):  # move to next organoid if no match
        continue

    # run PlatyMatch
    warnings.filterwarnings("ignore")
    ransac_iterations = 4000
    icp_iterations = 50

    print("matching of", R0_obj, "and", RX_obj, "in", plate_id, well_id)

    try:
        # affine_matches, ffd_matches = runPM(RX_numpy, R0_numpy, ransac_iterations, icp_iterations, RX_raw, R0_raw, RX_seg, R0_seg, "savename")

        affine_matches, transform_matrix_combined, confidence = runAffine(
            RX_numpy, R0_numpy, ransac_iterations, icp_iterations, save_images=False
        )

        affine_matches = pd.DataFrame(
            affine_matches, columns=["R0_nuc_id", "RX_nuc_id", "confidence"]
        )
        affine_matches["R0_organoid_id"] = R0_obj
        affine_matches["RX_organoid_id"] = RX_obj
        affine_matches["plate_id"] = plate_id
        affine_matches["well_id"] = well_id

        # Save measurement into the organoid directory.
        name = "linking_nuc_affine_" + names[1] + "to" + names[0]
        path = join(organoid.organoid_dir, name + ".csv")
        affine_matches.to_csv(path, index=False)  # saves csv

        # Add the measurement to the faim-hcs datastructure and save.
        organoid.add_measurement(name, path)
        organoid.save()  # updates json file

    except Exception as e:
        print(R0_obj, RX_obj, e)
        # print(R0_numpy.shape, RX_numpy.shape)

    try:
        ffd_matches = runFFD(
            RX_numpy,
            R0_numpy,
            ransac_iterations,
            icp_iterations,
            RX_raw,
            R0_raw,
            RX_seg,
            R0_seg,
            "savename",
            transform_matrix_combined,
            confidence,
            save_images=False,
        )

        ffd_matches = pd.DataFrame(ffd_matches, columns=["R0_nuc_id", "RX_nuc_id"])
        # ffd_matches = pd.DataFrame(ffd_matches, columns=['R0_nuc_id', 'RX_nuc_id', 'confidence'])
        ffd_matches["R0_organoid_id"] = R0_obj
        ffd_matches["RX_organoid_id"] = RX_obj
        ffd_matches["plate_id"] = plate_id
        ffd_matches["well_id"] = well_id

        # Save measurement into the organoid directory.
        name = "linking_nuc_ffd_" + names[1] + "to" + names[0]
        path = join(organoid.organoid_dir, name + ".csv")
        ffd_matches.to_csv(path, index=False)  # saves csv

        # Add the measurement to the faim-hcs datastructure and save.
        organoid.add_measurement(name, path)
        organoid.save()  # updates json file

    except Exception as e:
        print(R0_obj, RX_obj, e)
        # print(R0_numpy.shape, RX_numpy.shape)
