import argparse
import configparser
from typing import List

import prefect
from faim_hcs.hcs.Experiment import Experiment
from prefect import Flow, Parameter, task, unmapped
from prefect.executors import LocalDaskExecutor
from prefect.run_configs import LocalRun

from scmultiplex.linking.OrganoidLinking import get_linking_stats, link_organoids
from scmultiplex.utils.load_utils import load_experiment


@task(nout=2)
def load_exps(R0_path: str, RX_path: str):
    return load_experiment(R0_path), load_experiment(RX_path)


@task()
def get_names(RX_name: str):
    return ["R0", RX_name]


@task(nout=2)
def get_seg_and_folder_name(RX_name):
    return RX_name + "_linked", "obj_v0.3_registered_" + RX_name


@task()
def get_wells(exp: Experiment, excluded_plates: List[str]):
    exp.only_iterate_over_wells(True)
    exp.reset_iterator()
    wells = []
    for w in exp:
        if w.plate.plate_id in excluded_plates:
            continue  # skip plate

        wells.append(w)
    return wells


@task()
def link_organoids_task(well, ovr_channel, folder_name, R0, RX, seg_name, RX_name):
    link_organoids(
        well=well,
        ovr_channel=ovr_channel,
        folder_name=folder_name,
        R0=R0,
        RX=RX,
        seg_name=seg_name,
        RX_name=RX_name,
        logger=prefect.context.get("logger"),
    )

    return well


@task()
def get_linking_stats_task(well, seg_name, RX, iou_cutoff, names, ovr_channel):
    get_linking_stats(
        well=well,
        seg_name=seg_name,
        RX=RX,
        iou_cutoff=iou_cutoff,
        names=names,
        ovr_channel=ovr_channel,
        logger=prefect.context.get("logger"),
    )


with Flow(
    "Feature-Extraction",
    executor=LocalDaskExecutor(),
    run_config=LocalRun(),
) as flow:
    R0_dir = Parameter("R0_dir", default="/path/to/R0/summary.csv")
    RX_dir = Parameter("RX_dir", default="/path/to/RX/summary.csv")
    RX_name = Parameter("RX_name", default="R1")
    excluded_plates = Parameter("excluded_plates", default=[])
    iou_cutoff = Parameter("iou_cutoff", default=0.2)
    ovr_channel = Parameter("ovr_channel", default="C01")

    R0, RX = load_exps(R0_dir, RX_dir)
    names = get_names(RX_name)

    seg_name, folder_name = get_seg_and_folder_name(RX_name)

    wells = get_wells(R0, excluded_plates=excluded_plates)

    linked_wells = link_organoids_task.map(
        wells,
        unmapped(ovr_channel),
        unmapped(folder_name),
        unmapped(R0),
        unmapped(RX),
        unmapped(seg_name),
        unmapped(RX_name),
    )

    get_linking_stats_task.map(
        wells,
        unmapped(seg_name),
        unmapped(RX),
        unmapped(iou_cutoff),
        unmapped(names),
        unmapped(ovr_channel),
        upstream_tasks=[linked_wells],
    )


def conf_to_dict(config):
    return {
        "R0_dir": config["DEFAULT"]["R0_dir"],
        "RX_dir": config["DEFAULT"]["RX_dir"],
        "RX_name": config["DEFAULT"]["RX_name"],
        "excluded_plates": config["DEFAULT"]["excluded_plates"].split(","),
        "iou_cutoff": float(config["DEFAULT"]["iou_cutoff"]),
        "ovr_channel": config["DEFAULT"]["ovr_channel"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    kwargs = conf_to_dict(config)
    print(kwargs)

    flow.run(parameters=kwargs)


if __name__ == "__main__":
    main()
