import argparse
import configparser
from typing import List

import prefect
from faim_hcs.hcs.Experiment import Experiment
from prefect import Flow, Parameter, task, unmapped
from prefect.executors import LocalDaskExecutor
from prefect.run_configs import LocalRun

from scmultiplex.config import (
    commasplit,
    compute_workflow_params,
    get_round_names,
    get_workflow_params,
    summary_csv_path,
)
from scmultiplex.linking.OrganoidLinking import get_linking_stats, link_organoids
from scmultiplex.utils.exclude_utils import exclude_conditions
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
def get_wells(exp: Experiment, excluded_plates: List[str], excluded_wells: List[str]):
    return exclude_conditions(exp, excluded_plates, excluded_wells)


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
    excluded_wells = Parameter("excluded_wells", default=[])
    iou_cutoff = Parameter("iou_cutoff", default=0.2)
    ovr_channel = Parameter("ovr_channel", default="C01")

    R0, RX = load_exps(R0_dir, RX_dir)
    names = get_names(RX_name)

    seg_name, folder_name = get_seg_and_folder_name(RX_name)

    wells = get_wells(
        R0, excluded_plates=excluded_plates, excluded_wells=excluded_wells
    )

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

def get_config_params(config_file_path):
    
    round_names = get_round_names(config_file_path)
    if len(round_names) < 2:
        raise RuntimeError('At least two rounds are required to perform organoid linking')
    rounds_tobelinked = round_names[1:]
    config_params = {
        'ovr_channel':     ('01FeatureExtraction', 'ovr_channel'),
        }
    common_params = get_workflow_params(config_file_path, config_params)
    
    compute_param = {
        'excluded_plates': (
            commasplit,[
                ('01FeatureExtraction', 'excluded_plates')
                ]
            ),
        'excluded_wells': (
            commasplit,[
                ('01FeatureExtraction', 'excluded_wells')
                ]
            ),
        'iou_cutoff': (
            float,[
                ('02OrganoidLinking', 'iou_cutoff')
                ]
            ),
        'R0_dir': (
                summary_csv_path,[
                    ('00BuildExperiment', 'base_dir_save'),
                    ('00BuildExperiment.round_%s' % round_names[0], 'name')
                    ]
                ),
        }
    common_params.update(compute_workflow_params(config_file_path, compute_param))
    
    round_tobelinked_params = {}
    for ro in rounds_tobelinked:
        rp = common_params.copy()
        rp['RX_name'] = ro
        compute_param = {
            'RX_dir': (
                summary_csv_path,[
                    ('00BuildExperiment', 'base_dir_save'),
                    ('00BuildExperiment.round_%s' % ro, 'name')
                    ]
                ),
        }
        rp.update(compute_workflow_params(config_file_path, compute_param))
        round_tobelinked_params[ro] = rp
    return round_tobelinked_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()
    
    r_params = get_config_params(args.config)
    for ro, kwargs in r_params.items():
        flow.run(parameters = kwargs)


if __name__ == "__main__":
    main()
