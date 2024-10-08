#!/usr/bin/env python3

# Copyright (C) 2023 Friedrich Miescher Institute for Biomedical Research

##############################################################################
#                                                                            #
# Author: Nicole Repina              <nicole.repina@fmi.ch>                  #
# Author: Tim-Oliver Buchholz        <tim-oliver.buchholz@fmi.ch>            #
# Author: Enrico Tagliavini          <enrico.tagliavini@fmi.ch>              #
#                                                                            #
##############################################################################

import argparse
import os
import sys
from typing import List

import prefect
from prefect import Flow, Parameter, task, unmapped
from prefect.executors import LocalDaskExecutor
from prefect.run_configs import LocalRun

from scmultiplex import version
from scmultiplex.config import (
    commasplit,
    compute_workflow_params,
    get_round_names,
    get_workflow_params,
    summary_csv_path,
)
from scmultiplex.faim_hcs.hcs.Experiment import Experiment
from scmultiplex.linking.OrganoidLinking import get_linking_stats, link_organoids
from scmultiplex.logging import get_scmultiplex_logger, setup_prefect_handlers
from scmultiplex.utils import get_core_count
from scmultiplex.utils.exclude_utils import exclude_conditions
from scmultiplex.utils.load_utils import load_experiment


@task(nout=2)
def load_exps(R0_path: str, RX_path: str):
    return load_experiment(R0_path), load_experiment(RX_path)


@task()
def get_names(RX_name: str):
    return ["R0", RX_name]


@task(nout=2)
def get_seg_and_folder_name(RX_name, org_seg_name_RX):
    return RX_name + "_linked", org_seg_name_RX + "_registered_" + RX_name


@task()
def get_wells(exp: Experiment, excluded_plates: List[str], excluded_wells: List[str]):
    return exclude_conditions(exp, excluded_plates, excluded_wells)


@task()
def link_organoids_and_get_stats_task(
    well,
    org_seg_ch,
    folder_name,
    R0,
    RX,
    seg_name,
    mip_ovr_name_R0,
    mip_ovr_name_RX,
    iou_cutoff,
    names,
):
    link_organoids(
        well=well,
        ovr_channel=org_seg_ch,
        folder_name=folder_name,
        R0=R0,
        RX=RX,
        seg_name=seg_name,
        mip_ovr_name_R0=mip_ovr_name_R0,
        mip_ovr_name_RX=mip_ovr_name_RX,
        logger=get_scmultiplex_logger(),
    )
    get_linking_stats(
        well=well,
        seg_name=seg_name,
        RX=RX,
        iou_cutoff=iou_cutoff,
        names=names,
        ovr_channel=org_seg_ch,
        logger=get_scmultiplex_logger(),
    )
    return


# keep threading for time being
def run_flow(r_params, cpus):
    with Flow(
        "Feature-Extraction",
        executor=LocalDaskExecutor(scheduler="processes", num_workers=cpus),
        run_config=LocalRun(),
    ) as flow:
        R0_path = Parameter("R0_path")
        RX_path = Parameter("RX_path")
        RX_name = Parameter("RX_name")
        excluded_plates = Parameter("excluded_plates")
        excluded_wells = Parameter("excluded_wells")
        iou_cutoff = Parameter("iou_cutoff")
        org_seg_ch = Parameter("org_seg_ch")
        mip_ovr_name_R0 = Parameter("mip_ovr_name_R0")
        mip_ovr_name_RX = Parameter("mip_ovr_name_RX")
        org_seg_name_RX = Parameter("org_seg_name_RX")

        R0, RX = load_exps(R0_path, RX_path)
        names = get_names(RX_name)

        seg_name, folder_name = get_seg_and_folder_name(RX_name, org_seg_name_RX)

        wells = get_wells(
            R0, excluded_plates=excluded_plates, excluded_wells=excluded_wells
        )

        link_organoids_and_get_stats_task.map(
            wells,
            unmapped(org_seg_ch),
            unmapped(folder_name),
            unmapped(R0),
            unmapped(RX),
            unmapped(seg_name),
            unmapped(mip_ovr_name_R0),
            unmapped(mip_ovr_name_RX),
            unmapped(iou_cutoff),
            unmapped(names),
        )

    ret = 0
    for ro, kwargs in r_params.items():
        state = flow.run(parameters=kwargs)
        if state.is_failed():
            ret += 1
            break
    return ret


def get_config_params(config_file_path):

    round_names = get_round_names(config_file_path)
    if len(round_names) < 2:
        raise RuntimeError(
            "At least two rounds are required to perform organoid linking"
        )
    rounds_tobelinked = round_names[1:]

    config_params = {
        "mip_ovr_name_R0": (
            "00BuildExperiment.round_%s" % round_names[0],
            "mip_ovr_name",
        ),
    }

    common_params = get_workflow_params(config_file_path, config_params)

    compute_param = {
        "excluded_plates": (commasplit, [("01FeatureExtraction", "excluded_plates")]),
        "excluded_wells": (commasplit, [("01FeatureExtraction", "excluded_wells")]),
        "iou_cutoff": (float, [("02OrganoidLinking", "iou_cutoff")]),
        "R0_path": (
            summary_csv_path,
            [
                ("00BuildExperiment", "base_dir_save"),
                ("00BuildExperiment.round_%s" % round_names[0], "name"),
            ],
        ),
    }
    common_params.update(compute_workflow_params(config_file_path, compute_param))

    round_tobelinked_params = {}
    for ro in rounds_tobelinked:
        config_params = {
            "org_seg_ch": ("00BuildExperiment.round_%s" % ro, "organoid_seg_channel"),
            "mip_ovr_name_RX": ("00BuildExperiment.round_%s" % ro, "mip_ovr_name"),
            "org_seg_name_RX": ("00BuildExperiment.round_%s" % ro, "org_seg_name"),
        }
        rp = common_params.copy()
        rp.update(get_workflow_params(config_file_path, config_params))

        rp["RX_name"] = ro
        compute_param = {
            "RX_path": (
                summary_csv_path,
                [
                    ("00BuildExperiment", "base_dir_save"),
                    ("00BuildExperiment.round_%s" % ro, "name"),
                ],
            ),
        }
        rp.update(compute_workflow_params(config_file_path, compute_param))
        round_tobelinked_params[ro] = rp
    return round_tobelinked_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--cpus", type=int, default=get_core_count())
    parser.add_argument("--prefect-logfile", required=True)

    args = parser.parse_args()
    cpus = args.cpus
    prefect_logfile = args.prefect_logfile

    setup_prefect_handlers(prefect.utilities.logging.get_logger(), prefect_logfile)

    print("Running scMultipleX version %s" % version)

    r_params = get_config_params(args.config)

    ret = run_flow(r_params, cpus)
    if ret == 0:
        print("%s completed successfully" % os.path.basename(sys.argv[0]))
    return ret


if __name__ == "__main__":
    sys.exit(main())
