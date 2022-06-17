import os
from os.path import join

import pandas as pd
from faim_hcs.hcs.Experiment import Experiment

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)


# Load an existing faim-hcs Experiment from disk.
exp = Experiment()
exp.load(
    "/tungstenfs/scratch/gliberal/Users/repinico/Microscopy/Analysis/20220528_GCPLEX_redo/20220507GCPLEX_R0/summary.csv"
)

# pool together regionprops_org files
exp.only_iterate_over_wells(False)
exp.reset_iterator()

# create list of dataframes from all well and organoids
org_feat_df_list = (
    []
)  # length of list is #plates * # wells * #channels * # organoids/well

# add nuclear counts as well, if there is a nuc seg. this is only raw counts, before filtering or nuc/mem linking!
for organoid in exp:
    nuc_count = 0  # default is 0
    try:
        n = organoid.get_measurement("regionprops_nuc_C01")
    except Exception as e:
        print(
            organoid.organoid_id,
            organoid.well.well_id,
            organoid.well.plate.plate_id,
            "missing nuc seg",
            e,
        )  # usually exception is that no nuc were detected so csv is empty
    else:  # if there is no error, count number of nuclei in organoid
        if n is not None:
            nuc_count = n.shape[0]

    for meas_name in [
        k for k, v in organoid.measurements.items() if k.startswith("regionprops_org")
    ]:
        m = organoid.get_measurement(meas_name)
        m["nuc_count"] = nuc_count  # add nuc count to measurement
        org_feat_df_list.append(m)

org_feat_df = pd.concat(org_feat_df_list, ignore_index=True, sort=False)

# pool together regionprops_ovr files
exp.only_iterate_over_wells(True)
exp.reset_iterator()

# create list of ovr dataframes from all wells
org_ovr_df_list = []  # length of list is # plates * # wells

for well in exp:
    for meas_name in [
        k for k, v in well.measurements.items() if k.startswith("regionprops_ovr")
    ]:
        m = well.get_measurement(meas_name)
        org_ovr_df_list.append(m)

org_ovr_df = pd.concat(org_ovr_df_list, ignore_index=True, sort=False)

# merge all org and ovr features into single df, includes all plates, wells, and organoids

org_df = pd.merge(
    org_feat_df,
    org_ovr_df,
    how="left",
    on=["hcs_experiment", "plate_id", "well_id", "organoid_id"],
)
org_df = org_df.sort_values(
    by=["hcs_experiment", "root_dir", "plate_id", "well_id", "org_label", "channel_id"]
)

# add .fillna('NaN') at the end?

org_df.to_csv(join(exp.get_experiment_dir(), "org_df.csv"), index=False)  # saves csv

## Merge linking of organoids into single df - all rounds
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


## PlatyMatch linking into single df
# pool together organoid linking files
exp.only_iterate_over_wells(False)
exp.reset_iterator()

# create dictionary of lists containing linking files for each round
nuc_link_df_dict = {}  # each key in dictionary is a round that was linked to R0

# initialize dictionary, nested dictionary for each round
for organoid in exp:
    linkMeasurements = [
        k for k, v in organoid.measurements.items() if k.startswith("linking_nuc_ffd")
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
        k for k, v in organoid.measurements.items() if k.startswith("linking_nuc_ffd")
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
        nuc_link_df = pd.concat(nuc_link_df_dict[key], ignore_index=True, sort=False)
        nuc_link_df.to_csv(
            join(exp.get_experiment_dir(), ("linking_nuc_" + key + "_df.csv")),
            index=False,
        )  # saves csv


## Merge nuclear features into single df
# pool together regionprops_nuc files
exp.only_iterate_over_wells(False)
exp.reset_iterator()

# create list of dataframes from all well and organoids
nuc_feat_df_list = (
    []
)  # length of list is #plates * # wells * #channels * # organoids/well

for organoid in exp:
    nucMeasurements = [
        k for k, v in organoid.measurements.items() if k.startswith("regionprops_nuc")
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
            "org_label",
            "nuc_id",
            "channel_id",
        ]
    )
    nuc_df.to_csv(
        join(exp.get_experiment_dir(), "nuc_df.csv"), index=False
    )  # saves csv

## Merge membrane features into single df
# pool together regionprops_mem files
exp.only_iterate_over_wells(False)
exp.reset_iterator()

# create list of dataframes from all well and organoids
mem_feat_df_list = (
    []
)  # length of list is #plates * # wells * #channels * # organoids/well

for organoid in exp:
    memMeasurements = [
        k for k, v in organoid.measurements.items() if k.startswith("regionprops_mem")
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

# if there are any nuclear features, concatenate and save csv
if not len(mem_feat_df_list) == 0:
    mem_df = pd.concat(mem_feat_df_list, ignore_index=True, sort=False)
    mem_df = mem_df.sort_values(
        by=[
            "hcs_experiment",
            "root_dir",
            "plate_id",
            "well_id",
            "org_label",
            "mem_id",
            "channel_id",
        ]
    )
    mem_df.to_csv(
        join(exp.get_experiment_dir(), "mem_df.csv"), index=False
    )  # saves csv


## Link together nuclei and cells

# pool together regionprops_org files
exp.only_iterate_over_wells(False)
exp.reset_iterator()

# create list of dataframes from all well and organoids
cell_feat_df_list = []  # length of list is #plates * # wells * # organoids/well

for organoid in exp:
    nucMeasurements = [
        k for k, v in organoid.measurements.items() if k.startswith("regionprops_nuc")
    ]
    memMeasurements = [
        k for k, v in organoid.measurements.items() if k.startswith("regionprops_mem")
    ]
    linkMeasurement = [
        k
        for k, v in organoid.measurements.items()
        if k.startswith("linking_nuc_to_mem")
    ]

    # if no nuclear or membrane feature extraction in folder, skip. linking file must be unique
    if not nucMeasurements:
        continue
    if not memMeasurements:
        continue
    if not len(linkMeasurement) == 1:
        continue

    linking = organoid.get_measurement("linking_nuc_to_mem")
    linking_dict = linking.set_index("mem_id").T.to_dict("index")[
        "nuc_id"
    ]  # mem id is key, nuc id is value

    for meas_name in nucMeasurements:
        channel = meas_name[-3:]

        try:
            nuc = organoid.get_measurement(meas_name)
        except Exception as e:
            print(
                organoid.organoid_id,
                organoid.well.well_id,
                organoid.well.plate.plate_id,
                e,
            )  # usually exception is that no nuc were detected so csv is empty. in this case, skip organoid
            continue

        try:
            mem = organoid.get_measurement("regionprops_mem_" + channel)
        except Exception as e:
            print(
                organoid.organoid_id,
                organoid.well.well_id,
                organoid.well.plate.plate_id,
                e,
            )  # usually exception is that no nuc were detected so csv is empty. in this case, skip organoid
            continue

            # select nuclei that are matched to a membrane
        nuc_filt = nuc.loc[nuc["nuc_id"].isin(linking_dict.values()), :].copy(deep=True)
        nuc_filt = nuc_filt.rename(columns={"segmentation_nuc": "segmentation"})
        # select membranes that are matched to a nucleus
        mem_filt = mem.loc[mem["mem_id"].isin(linking_dict.keys()), :].copy(deep=True)
        mem_filt = mem_filt.rename(columns={"segmentation_mem": "segmentation"})

        # add column to nuc frame that includes nuc id
        nuc_filt["nuc_id_linked"] = nuc_filt["nuc_id"].astype(int)

        # add column to mem frame that includes matched nuc id
        mem_filt["nuc_id_linked"] = mem_filt["mem_id"].map(linking_dict).astype(int)

        # load matching cell file
        cell = pd.concat([nuc_filt, mem_filt], ignore_index=True)
        # append to list of dataframes
        cell_feat_df_list.append(cell)

# if there are cell features, concatenate and save csv
if not len(cell_feat_df_list) == 0:
    cell_df = pd.concat(cell_feat_df_list, ignore_index=True, sort=False)

    cell_df = cell_df.reset_index(drop=True)
    cols_to_move = [
        "hcs_experiment",
        "root_dir",
        "plate_id",
        "well_id",
        "org_label",
        "nuc_id_linked",
        "channel_id",
        "nuc_id",
        "mem_id",
    ]
    cell_df = cell_df[
        cols_to_move + [col for col in cell_df.columns if col not in cols_to_move]
    ]
    cell_df = cell_df.sort_values(
        by=[
            "hcs_experiment",
            "root_dir",
            "plate_id",
            "well_id",
            "org_label",
            "nuc_id_linked",
            "channel_id",
        ]
    )

    cell_df.to_csv(
        join(exp.get_experiment_dir(), "cell_df.csv"), index=False
    )  # saves csv


## Organoid linking over multiplexing rounds
exp_list = []
df_list = []

for i in range(len(round_names)):
    e = Experiment()
    e.load(round_directories[i])
    exp_list.append(e)

    path = os.path.join(e.get_experiment_dir(), "org_df.csv")
    isExist = os.path.exists(path)

    if not isExist:
        print("ERROR! Run org_df aggregation on round ", round_names[i])
        continue

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
            print(
                "ERROR! Run organoid linking and linking_org_df aggregation of round ",
                round_names[i],
            )
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

org_df = pd.DataFrame()
# select organoids that are matched to R1
for i in range(len(round_names) - 1):
    # i is 0
    right = dfs[round_names[i + 1]]  # load next round's df
    # right id is the organoid numbering relative to itself, i.e. RX
    right["id"] = (
        right["plate_id"]
        + "_"
        + right["well_id"]
        + "_"
        + right["organoid_id"].str.split("_").str[1]
    )  # right["org_label"].astype(str)
    link = links[round_names[i + 1]]  # load linking
    link_dict = link.set_index("id_RX").T.to_dict("index")[
        "id_R0"
    ]  # RX id is key, R0 is value

    # select right organoids that are matched to an R0 organoid
    # right_filt = right.copy(deep=True) #if do not want filtering, uncomment
    right_filt = right.loc[right["id"].isin(link_dict.keys()), :].copy(deep=True)

    right_filt["round_id"] = round_names[i + 1]  # add column with the round id
    right_filt["organoid_id_linked"] = right_filt["id"].map(link_dict)  # Link!

    if (
        i > 0
    ):  # need to also select organoids that are in the org_df (i.e. linked R0-R1 organoids)
        linked_organoids = (
            org_df["plate_id"]
            + "_"
            + org_df["well_id"]
            + "_"
            + org_df["org_label_linked"].astype(str)
        ).unique()
        print(linked_organoids)
        right_filt = right_filt.loc[
            right_filt["organoid_id_linked"].isin(linked_organoids), :
        ].copy(deep=True)

    right_filt["organoid_id_linked"] = (
        "object_" + right_filt["organoid_id_linked"].str.split("_").str[2]
    )
    right_filt["org_label_linked"] = (
        right_filt["organoid_id_linked"]
        .str.split("_")
        .str[1]
        .astype(float)
        .astype("Int64")
    )

    if i == 0:  # if linking R0-R1, use the R0 df
        left = dfs[round_names[0]]
        left["id"] = (
            left["plate_id"]
            + "_"
            + left["well_id"]
            + "_"
            + left["organoid_id"].str.split("_").str[1]
        )
        left["round_id"] = round_names[0]
        left["organoid_id_linked"] = left["organoid_id"]
        left["org_label_linked"] = (
            left["organoid_id_linked"].str.split("_").str[1].astype(int)
        )
        # select R0 organoids that are matched to the next round
        # to toggle, uncomment next line and comment subsequent
        left_filt = left.loc[left["id"].isin(link_dict.values()), :].copy(deep=True)
        # left_filt = left.copy(deep=True)

    # WARNING! DID NOT TEST MORE THAN TWO ROUNDS YET, CHECK THAT THIS WORKS
    else:  # if previous round linking already exists, use that df
        left = org_df.copy(deep=True)
        # update "id" to match R0 numbering
        left["id"] = (
            left["plate_id"]
            + "_"
            + left["well_id"]
            + "_"
            + left["organoid_id_linked"].str.split("_").str[1]
        )
        # filter organoids that missing matches in previous rounds;
        # so only have organoids that are linked across all rounds
        left_filt = left.loc[left["id"].isin(link_dict.values()), :].copy(deep=True)
        # if want organoids linked in R0-R1 but missing a link R0-R2, etc, uncomment next line
        # left_filt = left.copy(deep=True)

    org_df = pd.concat([left_filt, right_filt], ignore_index=True)

    # update sorting
    org_df = org_df.reset_index(drop=True)
    cols_to_move = ["plate_id", "well_id", "org_label_linked", "round_id"]
    org_df = org_df[
        cols_to_move + [col for col in org_df.columns if col not in cols_to_move]
    ]
    org_df = org_df.sort_values(by=cols_to_move)
    # multi-indexing
    # org_df = org_df.set_index(["plate_id","well_id", "org_label_linked", "round_id"]).sort_index()

# Save to R0 directory
org_df.to_csv(
    join(
        exps["R0"].get_experiment_dir(),
        ("org_df_linked_" + "-".join(round_names) + ".csv"),
    ),
    index=False,
)  # saves csv


## Nuclear linking over multiplexing rounds with PlatyMatch
# note all rounds must have the same plate names!
round_names = [
    "R0",
    "R1",
    "R2",
]  # ex. ['R0', 'R1', 'R2', 'R3'] always start with R0 and must be in sequential order

# #must be in same order as round_names above
# round_directories = ['/tungstenfs/scratch/gliberal/Users/repinico/Microscopy/Analysis/20220525_GCPLEX/20220507GCPLEX_R0/summary.csv',
#                      '/tungstenfs/scratch/gliberal/Users/repinico/Microscopy/Analysis/20220525_GCPLEX/20220507GCPLEX_R1/summary.csv',
#                     '/tungstenfs/scratch/gliberal/Users/repinico/Microscopy/Analysis/20220525_GCPLEX/20220507GCPLEX_R2/summary.csv']

# must be in same order as round_names above
round_directories = [
    "/tungstenfs/scratch/gliberal/Users/repinico/Microscopy/Analysis/20220528_GCPLEX_redo/20220507GCPLEX_R0/summary.csv",
    "/tungstenfs/scratch/gliberal/Users/repinico/Microscopy/Analysis/20220528_GCPLEX_redo/20220507GCPLEX_R1/summary.csv",
    "/tungstenfs/scratch/gliberal/Users/repinico/Microscopy/Analysis/20220528_GCPLEX_redo/20220507GCPLEX_R2/summary.csv",
]

exp_list = []
df_list = []

for i in range(len(round_names)):
    e = Experiment()
    e.load(round_directories[i])
    exp_list.append(e)

    path = os.path.join(e.get_experiment_dir(), "nuc_df.csv")
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

# dfs['R0']

link_names = []
link_list = []

for i in range(len(round_names)):
    # skip R0
    if i > 0:
        path = os.path.join(
            exps["R0"].get_experiment_dir(), ("linking_nuc_R" + str(i) + "_df.csv")
        )
        print(path)
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

nuc_df = pd.DataFrame()
# nuc_id_linked and organoid_id_linked is always the R0 numbering

for i in range(len(round_names) - 1):
    right = dfs[round_names[i + 1]]  # load next round's df
    right["id"] = (
        right["plate_id"]
        + "_"
        + right["well_id"]
        + "_"
        + right["organoid_id"].str.split("_").str[1]
        + "_"
        + right["nuc_id"].astype(int).astype(str)
    )
    link = links[round_names[i + 1]]  # load linking

    # discard all labels in RX that are not mapped to a single label in R0
    link["duplicated"] = link["id_RX"].duplicated()
    # if want to keep first duplicate of label, remove keep=False
    link_filtered = link[link["duplicated"] == False]
    print("removed", len(link) - len(link_filtered), "duplicated RX nuclei")

    link_dict = link_filtered.set_index("id_RX").T.to_dict("index")[
        "id_R0"
    ]  # RX id is key, R0 is value

    # select right organoids that are matched to R0 round
    # right_filt = right.copy(deep=True)
    right_filt = right.loc[right["id"].isin(link_dict.keys()), :].copy(deep=True)
    right_filt["round_id"] = round_names[i + 1]  # add column with the round id
    right_filt["nuc_id_linked"] = right_filt["id"].map(link_dict)  # link!
    # add R0 organoid id and label that the RX nuc is linked to
    right_filt["organoid_id_linked"] = (
        "object_" + right_filt["nuc_id_linked"].str.split("_").str[2]
    )
    right_filt["org_label_linked"] = (
        right_filt["organoid_id_linked"]
        .str.split("_")
        .str[1]
        .astype(float)
        .astype("Int64")
    )

    if (
        i > 0
    ):  # need to also select nuclei that are already in the nuc_df (i.e. linked R0-R1 nuclei)
        linked_nuclei = (
            nuc_df["plate_id"]
            + "_"
            + nuc_df["well_id"]
            + "_"
            + nuc_df["org_label_linked"].astype(int).astype(str)
            + "_"
            + nuc_df["nuc_id_linked"].astype(int).astype(str)
        ).unique()
        right_filt = right_filt.loc[
            right_filt["nuc_id_linked"].isin(linked_nuclei), :
        ].copy(deep=True)

    # add R0 nuclear label that the RX nuc is linked to
    right_filt["nuc_id_linked"] = (
        right_filt["nuc_id_linked"].str.split("_").str[3].astype(float).astype("Int64")
    )

    if i == 0:  # if linking R0-R1, use the R0 df
        left = dfs[round_names[0]]
        left["id"] = (
            left["plate_id"]
            + "_"
            + left["well_id"]
            + "_"
            + left["organoid_id"].str.split("_").str[1]
            + "_"
            + left["nuc_id"].astype(int).astype(str)
        )
        left["round_id"] = round_names[0]
        left["nuc_id_linked"] = left["nuc_id"].astype(int)
        left["organoid_id_linked"] = left["organoid_id"]
        left["org_label_linked"] = (
            left["organoid_id_linked"].str.split("_").str[1].astype(int)
        )
        # select R0 nuclei that are matched to the next round (different from organoid linking default!)
        # if want to change that all R0 nuc kept, comment next line and uncomment subsequent
        left_filt = left.loc[left["id"].isin(link_dict.values()), :].copy(deep=True)
        # left_filt = left.copy(deep=True)

    # WARNING! DID NOT TEST MORE THAN TWO ROUNDS YET, CHECK THAT THIS WORKS
    else:  # if previous round linking already exists, use that df
        left = nuc_df.copy(deep=True)
        # update "id" to match R0 numbering
        left["id"] = (
            left["plate_id"]
            + "_"
            + left["well_id"]
            + "_"
            + left["organoid_id_linked"].str.split("_").str[1]
            + "_"
            + left["nuc_id_linked"].astype(int).astype(str)
        )
        # select R0 nuclei that are matched to the RX round
        # if want to change that all R0 nuc kept, comment next line and uncomment subsequent
        # this means that output nuclei are successfully matched across all rounds
        left_filt = left.loc[left["id"].isin(link_dict.values()), :].copy(deep=True)
        # left_filt = left.copy(deep=True)

    nuc_df = pd.concat([left_filt, right_filt], ignore_index=True)

    # update sorting
    nuc_df = nuc_df.reset_index(drop=True)
    cols_to_move = [
        "plate_id",
        "well_id",
        "org_label_linked",
        "nuc_id_linked",
        "round_id",
    ]
    nuc_df = nuc_df[
        cols_to_move + [col for col in nuc_df.columns if col not in cols_to_move]
    ]
    nuc_df = nuc_df.sort_values(by=cols_to_move)

# Save to R0 directory
nuc_df.to_csv(
    join(
        exps["R0"].get_experiment_dir(),
        ("nuc_df_linked_" + "-".join(round_names) + ".csv"),
    ),
    index=False,
)  # saves csv

# multi-indexing
# nuc_df = nuc_df.set_index(["plate_id","well_id", "org_label_linked", "nuc_id_linked", "round_id"]).sort_index()
