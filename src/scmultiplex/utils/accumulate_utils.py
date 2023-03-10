# Copyright (C) 2023 Friedrich Miescher Institute for Biomedical Research

##############################################################################
#                                                                            #
# Author: Nicole Repina              <nicole.repina@fmi.ch>                  #
# Author: Tim-Oliver Buchholz        <tim-oliver.buchholz@fmi.ch>            #
# Author: Enrico Tagliavini          <enrico.tagliavini@fmi.ch>              #
#                                                                            #
##############################################################################

import os
from os.path import join

import numpy as np
import pandas as pd
from faim_hcs.hcs.Experiment import Experiment


def load_organoid_features(exp: Experiment):
    exp.only_iterate_over_wells(False)
    exp.reset_iterator()

    # create list of dataframes from all well and organoids
    org_feat_df_list = (
        []
    )  # length of list is #plates * # wells * #channels * # organoids/well

    # removed nuc count column because this is before filtering or nuc/mem linking!
    for organoid in exp:

        for meas_name in [
            k
            for k, v in organoid.measurements.items()
            if k.startswith("regionprops_org")
        ]:
            m = organoid.get_measurement(meas_name)
            org_feat_df_list.append(m)

    return pd.concat(org_feat_df_list, ignore_index=True, sort=False)


def load_well_features(exp: Experiment):
    exp.only_iterate_over_wells(True)
    exp.reset_iterator()

    # create list of ovr dataframes from all wells
    org_ovr_df_list = []  # length of list is # plates * # wells

    for well in exp:
        for meas_name in [
            k for k, v in well.measurements.items() if k.startswith("regionprops_ovr")
        ]:
            try:
                m = well.get_measurement(meas_name)
                org_ovr_df_list.append(m)

            except Exception as e:
                print(
                    well.well_id, well.plate.plate_id, "missing ovr", e
                )  # usually exception is that no nuc were detected so csv is empty

    return pd.concat(org_ovr_df_list, ignore_index=True, sort=False)


def accumulate_tables(exp: Experiment):
    org_feat_df = load_organoid_features(exp)
    org_ovr_df = load_well_features(exp)

    org_df = pd.merge(
        org_feat_df,
        org_ovr_df,
        how="left",
        on=["hcs_experiment", "plate_id", "well_id", "org_id"],
    )
    org_df = org_df.sort_values(
        by=[
            "hcs_experiment",
            "root_dir",
            "plate_id",
            "well_id",
            "org_id",
            "channel_id",
        ]
    )

    if org_df is None:
        raise RuntimeWarning("No organoid feature extraction found in %s" % exp.name)

    return org_df


def split_shape_vs_channel_feats(df,tidy_id):
    """identify object (shape) features vs. channel features in columns of df
    """
    # split by tidy id; each group is a nucleus
    splits =df.groupby(tidy_id)

    # for each nuc, calculate how many unique values of each measurement there are
    # take the max over all nuclei
    # col_classes = splits.aggregate(lambda x: len(np.unique(x))).max()
    col_classes = splits.aggregate(lambda x: len(x.dropna().unique())).max()

    # if this value is equal to the number of channels in this round, then the value is repeated
    # else value can have unique value per channel
    n = len(np.unique(df["channel_id"]))

    # classify colnames as object or channel features
    # False if different measurements over channels (e.g. intensity feats), channel properties;
    # True if all measurements have the same measurement (e.g.shape feats), object properties
    col_classes = col_classes.apply(lambda x: False if x == n else True)
    col_classes = col_classes.to_dict()

    all_feats = np.array(df.columns)
    all_feats = all_feats[
        all_feats != tidy_id
    ]  # list of all col names of df except tidy_id

    col_booleans = np.array(
        [col_classes[k] for k in all_feats]
    )  # list of booleans according to dict

    object_feats = all_feats[col_booleans]  # col names of object features (True)
    channel_feats = all_feats[~col_booleans]  # col names of channel features (False)
    return object_feats, channel_feats


def save_tidy_plate_well_org(exp: Experiment, seg_channels):
    org_seg_ch = seg_channels[0]
    org_df = accumulate_tables(exp)

    tidy_id = "plate_well_org"
    org_df[tidy_id] = (
        org_df["plate_id"]
        + "_"
        + org_df["well_id"]
        + "_"
        + org_df["org_id"].astype(str)
    )

    # identify object (shape) features vs. channel features
    object_feats, channel_feats = split_shape_vs_channel_feats(org_df, tidy_id)

    # make dataframe where each row is single organoid
    # if column is the same for all channels (True), its first value is taken and placed into dataframe
    # if column is different across channels (False), make column with that feature name plus channel ID

    splits = org_df.groupby("channel_id")

    org_df_tidy = splits.get_group(org_seg_ch).copy().reset_index(drop=True)
    org_df_tidy = org_df_tidy.set_index(tidy_id)
    org_df_tidy = org_df_tidy[object_feats]  # select only object features
    org_df_tidy = org_df_tidy.add_prefix("C00.org.")

    # then add on columns for channel-specific features

    for ch in np.unique(org_df["channel_id"]):
        org_df_ch = splits.get_group(ch).reset_index(drop=True)
        org_df_ch = org_df_ch.set_index(tidy_id)
        # select channel columns
        org_df_ch = org_df_ch[channel_feats]
        org_df_ch = org_df_ch.drop(columns="channel_id")
        # remove channel_id column

        org_df_ch = org_df_ch.add_prefix(ch + ".org.")

        org_df_tidy = org_df_tidy.join(org_df_ch)

    org_df_tidy.to_csv(
        join(exp.get_experiment_dir(), "org_df_tidy.csv"), index=False
    )  # saves csv


def load_nuclei_features(exp: Experiment):
    exp.only_iterate_over_wells(False)
    exp.reset_iterator()

    # create list of dataframes from all well and organoids
    nuc_feat_df_list = (
        []
    )  # length of list is #plates * # wells * #channels * # organoids/well

    for organoid in exp:
        nucMeasurements = [
            k
            for k, v in organoid.measurements.items()
            if k.startswith("regionprops_nuc")
        ]

        # if no nuclear feature extraction in folder, skip
        if not nucMeasurements:
            continue

        for meas_name in nucMeasurements:
            try:
                m = organoid.get_measurement(meas_name)

            except Exception as e:
                print(
                    organoid.organoid_id,
                    organoid.well.well_id,
                    organoid.well.plate.plate_id,
                    e,
                )  # usually exception is that no nuc were detected so csv is empty. in this case, skip organoid
                continue

            nuc_feat_df_list.append(m)

    # if there are any nuclear features, concatenate and save csv
    if not len(nuc_feat_df_list) == 0:
        nuc_df = pd.concat(nuc_feat_df_list, ignore_index=True, sort=False)

        nuc_df = nuc_df.sort_values(
            by=[
                "hcs_experiment",
                "root_dir",
                "plate_id",
                "well_id",
                "org_id",
                "nuc_id",
                "channel_id",
            ]
        )

        return nuc_df
    raise RuntimeWarning("No nuclear feature extraction found in %s" % exp.name)


def save_tidy_plate_well_org_nuc(exp: Experiment, seg_channels):
    nuc_seg_ch = seg_channels[1]
    nuc_df = load_nuclei_features(exp)

    # make tidy id
    tidy_id = "plate_well_org_nuc"
    nuc_df[tidy_id] = (
        nuc_df["plate_id"]
        + "_"
        + nuc_df["well_id"]
        + "_"
        + nuc_df["org_id"].astype(str)
        + "_"
        + nuc_df["nuc_id"].astype(str)
    )

    # take 100 random nuclei and determine feature classifications from them ; otherwise very slow
    npoints = 100

    tidy_ids = np.unique(nuc_df[tidy_id])

    # select random set of nuclei
    if len(nuc_df) > npoints:
        subset = np.random.choice(tidy_ids, size=npoints, replace=False)
    else:
        subset = tidy_ids

    # select data for this random set of nuclei
    nuc_df_subset = nuc_df.loc[(nuc_df[tidy_id].isin(subset))]

    # identify object (shape) features vs. channel features
    object_feats, channel_feats = split_shape_vs_channel_feats(nuc_df_subset, tidy_id)

    # make dataframe where each row is single organoid
    # if column is the same for all channels (True), its first value is taken and placed into dataframe
    # if column is different across channels (False), make column with that feature name plus channel ID

    splits = nuc_df.groupby("channel_id")

    nuc_df_tidy = splits.get_group(nuc_seg_ch).copy().reset_index(drop=True)
    nuc_df_tidy = nuc_df_tidy.set_index(tidy_id)
    nuc_df_tidy = nuc_df_tidy[object_feats]  # select only object features
    nuc_df_tidy = nuc_df_tidy.add_prefix("C00.nuc.")

    # then add on columns for channel-specific features

    for ch in np.unique(nuc_df["channel_id"]):
        nuc_df_ch = splits.get_group(ch).reset_index(drop=True)
        nuc_df_ch = nuc_df_ch.set_index(tidy_id)
        # select channel columns
        nuc_df_ch = nuc_df_ch[channel_feats]
        nuc_df_ch = nuc_df_ch.drop(columns="channel_id")
        # remove channel_id column

        nuc_df_ch = nuc_df_ch.add_prefix(ch + ".nuc.")

        nuc_df_tidy = nuc_df_tidy.join(nuc_df_ch)

    nuc_df_tidy.to_csv(
        join(exp.get_experiment_dir(), "nuc_df_tidy.csv"), index=False
    )  # saves csv


def load_membrane_features(exp: Experiment):
    # note: this is identical to the nuclear aggregate above, only for membrane seg
    # only difference is aggregation is run with mem_df that has mem_id column, and the 'mem' prefix is added to output (mem_df_tidy) column names
    # pool together regionprops_nuc files
    exp.only_iterate_over_wells(False)
    exp.reset_iterator()

    # create list of dataframes from all well and organoids
    mem_feat_df_list = (
        []
    )  # length of list is #plates * # wells * #channels * # organoids/well

    for organoid in exp:
        memMeasurements = [
            k
            for k, v in organoid.measurements.items()
            if k.startswith("regionprops_mem")
        ]

        # if no nuclear feature extraction in folder, skip
        if not memMeasurements:
            continue

        for meas_name in memMeasurements:
            try:
                m = organoid.get_measurement(meas_name)

            except Exception as e:
                print(
                    organoid.organoid_id,
                    organoid.well.well_id,
                    organoid.well.plate.plate_id,
                    e,
                )  # usually exception is that no nuc were detected so csv is empty. in this case, skip organoid
                continue

            mem_feat_df_list.append(m)

    # if there are any membrane features, concatenate and save csv
    if not len(mem_feat_df_list) == 0:
        mem_df = pd.concat(mem_feat_df_list, ignore_index=True, sort=False)
        mem_df = mem_df.sort_values(
            by=[
                "hcs_experiment",
                "root_dir",
                "plate_id",
                "well_id",
                "org_id",
                "mem_id",
                "channel_id",
            ]
        )
        return mem_df
    raise RuntimeWarning("No membrane feature extraction found in %s" % exp.name)


def save_tidy_plate_well_org_mem(exp: Experiment, seg_channels):
    mem_seg_ch = seg_channels[2]
    mem_df = load_membrane_features(exp)

    # make tidy id
    tidy_id = "plate_well_org_mem"
    mem_df[tidy_id] = (
        mem_df["plate_id"]
        + "_"
        + mem_df["well_id"]
        + "_"
        + mem_df["org_id"].astype(str)
        + "_"
        + mem_df["mem_id"].astype(str)
    )

    # take 100 random membranes and determine feature classifications from them ; otherwise very slow
    npoints = 100

    tidy_ids = np.unique(mem_df[tidy_id])

    # select random set of nuclei
    if len(mem_df) > npoints:
        subset = np.random.choice(tidy_ids, size=npoints, replace=False)
    else:
        subset = tidy_ids

    # select data for this random set of nuclei
    mem_df_subset = mem_df.loc[(mem_df[tidy_id].isin(subset))]

    # identify object (shape) features vs. channel features
    object_feats, channel_feats = split_shape_vs_channel_feats(mem_df_subset, tidy_id)

    # make dataframe where each row is single organoid
    # if column is the same for all channels (True), its first value is taken and placed into dataframe
    # if column is different across channels (False), make column with that feature name plus channel ID

    splits = mem_df.groupby("channel_id")

    mem_df_tidy = splits.get_group(mem_seg_ch).copy().reset_index(drop=True)
    mem_df_tidy = mem_df_tidy.set_index(tidy_id)
    mem_df_tidy = mem_df_tidy[object_feats]  # select only object features
    mem_df_tidy = mem_df_tidy.add_prefix("C00.mem.")

    # then add on columns for channel-specific features

    for ch in np.unique(mem_df["channel_id"]):
        mem_df_ch = splits.get_group(ch).reset_index(drop=True)
        mem_df_ch = mem_df_ch.set_index(tidy_id)
        # select channel columns
        mem_df_ch = mem_df_ch[channel_feats]
        mem_df_ch = mem_df_ch.drop(columns="channel_id")
        # remove channel_id column

        mem_df_ch = mem_df_ch.add_prefix(ch + ".mem.")

        mem_df_tidy = mem_df_tidy.join(mem_df_ch)

    mem_df_tidy.to_csv(
        join(exp.get_experiment_dir(), "mem_df_tidy.csv"), index=False
    )  # saves csv


def write_nuc_to_mem_linking(exp: Experiment):
    exp.only_iterate_over_wells(False)
    exp.reset_iterator()

    # create df that aggregates all of the per-organoid linking csv's into single file that is saved in faim-hcs exp home directory
    linking_list = []

    # load linking files
    for organoid in exp:

        linkMeasurement = [
            k
            for k, v in organoid.measurements.items()
            if k.startswith("linking_nuc_to_mem")
        ]

        # if no nuclear or membrane feature extraction in folder, skip. linking file must be unique
        if not linkMeasurement:
            continue
        if not len(linkMeasurement) == 1:
            continue

        linking = organoid.get_measurement("linking_nuc_to_mem")
        linking_list.append(linking)

    if len(linking_list) > 0:
        # aggregate linking files and save
        link_df = pd.concat(linking_list, ignore_index=True, sort=False)
        link_df.to_csv(
            join(exp.get_experiment_dir(), "linking_nuc_to_mem.csv"), index=False
        )  # saves csv
    else:
        raise RuntimeWarning(
            "No nuclear to membrane linking found for experiment %s" % exp.name
        )


def write_merged_nuc_membrane_features(exp: Experiment):
    # load data
    base = exp.get_experiment_dir()

    # load organoid dataframe
    nuc_df_tidy = pd.read_csv(os.path.join(base, "nuc_df_tidy.csv"))
    mem_df_tidy = pd.read_csv(os.path.join(base, "mem_df_tidy.csv"))
    link_df = pd.read_csv(os.path.join(base, "linking_nuc_to_mem.csv"))

    # make linking id's

    nuc_df_tidy["nID"] = (
        nuc_df_tidy["C00.nuc.plate_id"]
        + "_"
        + nuc_df_tidy["C00.nuc.well_id"]
        + "_"
        + nuc_df_tidy["C00.nuc.org_id"].astype(str)
        + "_"
        + nuc_df_tidy["C00.nuc.nuc_id"].astype(str)
    )
    mem_df_tidy["mID"] = (
        mem_df_tidy["C00.mem.plate_id"]
        + "_"
        + mem_df_tidy["C00.mem.well_id"]
        + "_"
        + mem_df_tidy["C00.mem.org_id"].astype(str)
        + "_"
        + mem_df_tidy["C00.mem.mem_id"].astype(str)
    )

    link_df["nID"] = (
        link_df["plate_id"]
        + "_"
        + link_df["well_id"]
        + "_"
        + link_df["org_id"].astype(str)
        + "_"
        + link_df["nuc_id"].astype(str)
    )
    link_df["mID"] = (
        link_df["plate_id"]
        + "_"
        + link_df["well_id"]
        + "_"
        + link_df["org_id"].astype(str)
        + "_"
        + link_df["mem_id"].astype(str)
    )

    # create dictionary
    # membrane id is key; linked nucleus is value
    link_dict = link_df.set_index("mID").T.to_dict("index")["nID"]

    mem_df_tidy["LINKED-nID"] = mem_df_tidy["mID"].map(link_dict)  # Link!

    # join the nuc and mem df's
    nuc_df_tidy = nuc_df_tidy.set_index("nID")
    mem_df_tidy = mem_df_tidy.set_index(
        "LINKED-nID"
    )  # NaN can be a row name if no nucleus is linked to this membrane; however NaN row names are ignored during join so are discarded
    mem_df_tidy = mem_df_tidy.drop(columns=["mID"])

    cell_df_tidy = nuc_df_tidy.join(mem_df_tidy, how="inner")  # join!

    # update indexing
    cell_df_tidy = cell_df_tidy.reset_index(drop=True)

    # save
    cell_df_tidy.to_csv(
        join(exp.get_experiment_dir(), "cell_df_tidy.csv"), index=False
    )  # saves csv


def merge_org_linking(exp: Experiment):
    # pool together organoid linking files
    exp.only_iterate_over_wells(True)
    exp.reset_iterator()

    # create dictionary of lists containing linking files for each round
    org_link_df_dict = {}  # each key in dictionary is a round that was linked to R0

    # initialize dictionary
    for well in exp:
        linkMeasurements = [
            k for k, v in well.measurements.items() if k.startswith("linking_ovr")
        ]

        # if no linking in folder, skip
        if not linkMeasurements:
            continue

        for meas_name in linkMeasurements:
            rnd = meas_name[-6:-4]  # select the round name that was linked to R0
            org_link_df_dict[rnd] = []  # add to dictionary as empty list

    # add linking files to dictionary
    exp.only_iterate_over_wells(True)
    exp.reset_iterator()
    for well in exp:
        linkMeasurements = [
            k for k, v in well.measurements.items() if k.startswith("linking_ovr")
        ]

        # if no linking in folder, skip
        if not linkMeasurements:
            continue

        for meas_name in linkMeasurements:
            m = well.get_measurement(meas_name)
            rnd = meas_name[-6:-4]
            org_link_df_dict[rnd].append(m)

    # if there is organoid linking, concatenate and save csv
    if not len(org_link_df_dict) == 0:
        for key in org_link_df_dict:
            link_df = pd.concat(org_link_df_dict[key], ignore_index=True, sort=False)
            link_df.to_csv(
                join(exp.get_experiment_dir(), ("linking_org_" + key + "_df.csv")),
                index=False,
            )  # saves csv


def merge_platymatch_linking(exp: Experiment):
    # pool together organoid linking files
    exp.only_iterate_over_wells(False)
    exp.reset_iterator()

    # create dictionary of lists containing linking files for each round
    nuc_link_df_dict = {}  # each key in dictionary is a round that was linked to R0

    # initialize dictionary, nested dictionary for each round
    for organoid in exp:
        linkMeasurements = [
            k
            for k, v in organoid.measurements.items()
            if k.startswith("linking_nuc_ffd")
        ]

        # if no linking in organoid, skip
        if not linkMeasurements:
            continue

        for meas_name in linkMeasurements:
            rnd = meas_name[-6:-4]  # select the round name that was linked to R0
            nuc_link_df_dict[rnd] = []  # add to dictionary as empty list

    # add linking files to dictionary
    exp.only_iterate_over_wells(False)
    exp.reset_iterator()
    for organoid in exp:
        linkMeasurements = [
            k
            for k, v in organoid.measurements.items()
            if k.startswith("linking_nuc_ffd")
        ]

        # if no linking in organoid, skip
        if not linkMeasurements:
            continue

        for meas_name in linkMeasurements:
            m = organoid.get_measurement(meas_name)
            rnd = meas_name[-6:-4]
            nuc_link_df_dict[rnd].append(m)

    # if there is nuc linking, concatenate and save csv
    if not len(nuc_link_df_dict) == 0:
        for key in nuc_link_df_dict:
            nuc_link_df = pd.concat(
                nuc_link_df_dict[key], ignore_index=True, sort=False
            )
            nuc_link_df.to_csv(
                join(exp.get_experiment_dir(), ("linking_nuc_" + key + "_df.csv")),
                index=False,
            )  # saves csv


def write_organoid_linking_over_multiplexing_rounds(round_names, round_summary_csvs):
    # Load the data

    exp_list = []
    df_list = []

    for i in range(len(round_names)):
        e = Experiment()
        e.load(round_summary_csvs[i])
        exp_list.append(e)

        path = os.path.join(e.get_experiment_dir(), "org_df_tidy.csv")
        assert os.path.exists(path), (
            "ERROR! Run org_df aggregation on round " + round_names[i]
        )

        # load organoid dataframe
        df = pd.read_csv(path)
        df_list.append(df)

    # create dictionary with keys round names, values experiment objects or org_dfs
    exps = dict(zip(round_names, exp_list))
    dfs = dict(zip(round_names, df_list))

    # dfs['R0']

    link_names = []
    link_list = []

    for i in range(len(round_names)):
        if i > 0:
            path = os.path.join(
                exps["R0"].get_experiment_dir(), ("linking_org_R" + str(i) + "_df.csv")
            )
            isExist = os.path.exists(path)

            if not isExist:
                print("ERROR! Run linking_org_df aggregation of round ", round_names[i])
                continue

            # load linking dataframe
            df = pd.read_csv(path)
            df["id_R0"] = (
                df["plate_id"] + "_" + df["well_id"] + "_" + df["R0_label"].astype(str)
            )
            df["id_RX"] = (
                df["plate_id"] + "_" + df["well_id"] + "_" + df["RX_label"].astype(str)
            )
            link_names.append(round_names[i])
            link_list.append(df)

    # create dictionary with keys round names, values linking dfs
    links = dict(zip(link_names, link_list))

    # Link all rounds to round 0 numbering
    org_df_tidy = pd.DataFrame()

    tidy_id = "plate_well_org_linked"

    # select organoids that are matched to R1
    for i in range(len(round_names) - 1):

        # Prepare 'right' df:
        # i is 0
        right = dfs[round_names[i + 1]]  # load next round's df
        # right id is the organoid numbering relative to itself, i.e. RX
        right["C00.org.LINK"] = (
            right["C00.org.plate_id"]
            + "_"
            + right["C00.org.well_id"]
            + "_"
            + right["C00.org.org_id"].astype(str)
        )
        link = links[round_names[i + 1]]  # load linking
        link_dict = link.set_index("id_RX").T.to_dict("index")[
            "id_R0"
        ]  # RX id is key, R0 is value

        right["C00.org.LINK-MAPPED"] = right["C00.org.LINK"].map(link_dict)  # Link!
        right["C00.org.LINK-MAPPED-ID"] = (
            right["C00.org.LINK-MAPPED"]
            .str.split("_")
            .str[2]
            .astype(float)
            .astype("Int64")
        )

        right[tidy_id] = (
            right["C00.org.plate_id"]
            + "_"
            + right["C00.org.well_id"]
            + "_"
            + right["C00.org.LINK-MAPPED-ID"].astype(str)
        )
        right = right.set_index(tidy_id)
        # remove redudant columns
        right = right.drop(
            columns=[
                "C00.org.plate_id",
                "C00.org.well_id",
                "C00.org.object_type",
                "C00.org.LINK",
                "C00.org.LINK-MAPPED",
                "C00.org.LINK-MAPPED-ID",
            ]
        )
        # add round prefix to remaining cols
        right = right.add_prefix(
            round_names[i + 1] + "."
        )  # adds prefix to all except for index

        # Prepare 'left' df:
        if i == 0:  # if linking R0-R1, use the R0 df
            left = dfs[round_names[0]]

            left[tidy_id] = (
                left["C00.org.plate_id"]
                + "_"
                + left["C00.org.well_id"]
                + "_"
                + left["C00.org.org_id"].astype(str)
            )
            left = left.set_index(tidy_id)
            # remove redudant columns
            left = left.drop(columns=["C00.org.object_type"])
            # rename key columns so they are agnostic to round
            left = left.add_prefix(
                round_names[0] + "."
            )  # adds prefix to all except for index
            left = left.rename(
                columns={
                    round_names[0] + "." + "C00.org.plate_id": "plate_id",
                    round_names[0] + "." + "C00.org.well_id": "well_id",
                    round_names[0] + "." + "C00.org.org_id": "org_id",
                }
            )

        else:  # if previous round linking already exists, use that df
            left = org_df_tidy.copy(deep=True)

        # change 'how' parameter if want organoids linked in R0-R1 but missing a link R0-R2, etc.
        org_df_tidy = left.join(right, how="inner")

    # update sorting
    org_df_tidy = org_df_tidy.reset_index(drop=True)
    cols_to_move = ["plate_id", "well_id", "org_id"]
    org_df_tidy = org_df_tidy[
        cols_to_move + [col for col in org_df_tidy.columns if col not in cols_to_move]
    ]
    org_df_tidy = org_df_tidy.sort_values(by=cols_to_move)

    # Save to R0 directory
    org_df_tidy.to_csv(
        join(
            exps["R0"].get_experiment_dir(),
            ("org_df_tidy_linked_" + "-".join(round_names) + ".csv"),
        ),
        index=False,
    )  # saves csv


def write_nuclear_linking_over_multiplexing_rounds(round_names, round_summary_csvs):
    # Load the data
    exp_list = []
    df_list = []

    for i in range(len(round_names)):
        e = Experiment()
        e.load(round_summary_csvs[i])
        exp_list.append(e)

        path = os.path.join(e.get_experiment_dir(), "nuc_df_tidy.csv")
        isExist = os.path.exists(path)

        if not isExist:
            print("ERROR! Run nuc_df aggregation on round ", round_names[i])
            continue

        # load nuc dataframe
        df = pd.read_csv(path)
        df_list.append(df)

    # create dictionary with keys round names, values experiment objects or nuc_dfs
    exps = dict(zip(round_names, exp_list))
    dfs = dict(zip(round_names, df_list))

    link_names = []
    link_list = []

    for i in range(len(round_names)):
        # skip R0
        if i > 0:
            path = os.path.join(
                exps["R0"].get_experiment_dir(), ("linking_nuc_R" + str(i) + "_df.csv")
            )
            isExist = os.path.exists(path)

            if not isExist:
                print(
                    "ERROR! Run organoid linking and linking_nuc_df aggregation of round ",
                    round_names[i],
                )
                continue

            # load linking dataframe
            df = pd.read_csv(path)
            df["id_R0"] = (
                df["plate_id"]
                + "_"
                + df["well_id"]
                + "_"
                + df["R0_organoid_id"].str.split("_").str[1]
                + "_"
                + df["R0_nuc_id"].astype(int).astype(str)
            )
            df["id_RX"] = (
                df["plate_id"]
                + "_"
                + df["well_id"]
                + "_"
                + df["RX_organoid_id"].str.split("_").str[1]
                + "_"
                + df["RX_nuc_id"].astype(int).astype(str)
            )
            link_names.append(round_names[i])
            link_list.append(df)

    # create dictionary with keys round names, values linking dfs
    links = dict(zip(link_names, link_list))

    # Link all rounds to round 0 numbering

    nuc_df_tidy = pd.DataFrame()
    # nuc_id_linked and organoid_id_linked is always the R0 numbering

    tidy_id = "plate_well_org_nuc_linked"

    for i in range(len(round_names) - 1):
        right = dfs[round_names[i + 1]]  # load next round's df
        # construct id used for linking
        right["C00.nuc.LINK"] = (
            right["C00.nuc.plate_id"]
            + "_"
            + right["C00.nuc.well_id"]
            + "_"
            + right["C00.nuc.org_id"].astype(str)
            + "_"
            + right["C00.nuc.nuc_id"].astype(str)
        )
        link = links[round_names[i + 1]]  # load linking

        # discard all labels in RX that are not mapped to a single label in R0
        link["duplicated"] = link["id_RX"].duplicated()
        # if want to keep first duplicate of label, remove keep=False
        link_filtered = link[[not elem for elem in link["duplicated"]]]
        # print("removed", len(link) - len(link_filtered), "duplicated RX nuclei")

        link_dict = link_filtered.set_index("id_RX").T.to_dict("index")[
            "id_R0"
        ]  # RX id is key, R0 is value

        #     select right organoids that are matched to R0 round - not really necessary but neater
        right = right.loc[right["C00.nuc.LINK"].isin(link_dict.keys()), :].copy(
            deep=True
        )

        right["C00.nuc.LINK-MAPPED"] = right["C00.nuc.LINK"].map(link_dict)  # link!
        # add R0 organoid id and label that the RX nuc is linked to
        right["C00.nuc.LINK-MAPPED-ORGID"] = (
            right["C00.nuc.LINK-MAPPED"]
            .str.split("_")
            .str[2]
            .astype(float)
            .astype("Int64")
        )
        # add R0 nuclear label that the RX nuc is linked to
        right["C00.nuc.LINK-MAPPED-NUCID"] = (
            right["C00.nuc.LINK-MAPPED"]
            .str.split("_")
            .str[3]
            .astype(float)
            .astype("Int64")
        )

        right[tidy_id] = (
            right["C00.nuc.plate_id"]
            + "_"
            + right["C00.nuc.well_id"]
            + "_"
            + right["C00.nuc.LINK-MAPPED-ORGID"].astype(str)
            + "_"
            + right["C00.nuc.LINK-MAPPED-NUCID"].astype(str)
        )
        right = right.set_index(tidy_id)
        # remove redudant columns
        right = right.drop(
            columns=[
                "C00.nuc.plate_id",
                "C00.nuc.well_id",
                "C00.nuc.object_type",
                "C00.nuc.LINK",
                "C00.nuc.LINK-MAPPED",
                "C00.nuc.LINK-MAPPED-ORGID",
                "C00.nuc.LINK-MAPPED-NUCID",
            ]
        )

        # add round prefix to remaining cols
        right = right.add_prefix(
            round_names[i + 1] + "."
        )  # adds prefix to all except for index

        if (
            i == 0
        ):  # if linking R0-R1, use the R0 df; include here all organoids identified in R0
            left = dfs[round_names[0]]
            left[tidy_id] = (
                left["C00.nuc.plate_id"]
                + "_"
                + left["C00.nuc.well_id"]
                + "_"
                + left["C00.nuc.org_id"].astype(str)
                + "_"
                + left["C00.nuc.nuc_id"].astype(str)
            )
            left = left.set_index(tidy_id)
            # remove redudant columns
            left = left.drop(columns=["C00.nuc.object_type"])
            # rename key columns so they are agnostic to round
            left = left.add_prefix(
                round_names[0] + "."
            )  # adds prefix to all except for index
            left = left.rename(
                columns={
                    round_names[0] + "." + "C00.nuc.plate_id": "plate_id",
                    round_names[0] + "." + "C00.nuc.well_id": "well_id",
                    round_names[0] + "." + "C00.nuc.org_id": "org_id",
                    round_names[0] + "." + "C00.nuc.nuc_id": "nuc_id",
                }
            )

        else:  # if previous round linking already exists, use that df
            left = nuc_df_tidy.copy(deep=True)

        # change 'how' parameter if want organoids linked in R0-R1 but missing a link R0-R2, etc.
        nuc_df_tidy = left.join(right, how="inner")

    # update sorting
    nuc_df_tidy = nuc_df_tidy.reset_index(drop=True)
    cols_to_move = ["plate_id", "well_id", "org_id", "nuc_id"]
    nuc_df_tidy = nuc_df_tidy[
        cols_to_move + [col for col in nuc_df_tidy.columns if col not in cols_to_move]
    ]
    nuc_df_tidy = nuc_df_tidy.sort_values(by=cols_to_move)

    # Save to R0 directory
    nuc_df_tidy.to_csv(
        join(
            exps["R0"].get_experiment_dir(),
            ("nuc_df_tidy_linked_" + "-".join(round_names) + ".csv"),
        ),
        index=False,
    )  # saves csv
