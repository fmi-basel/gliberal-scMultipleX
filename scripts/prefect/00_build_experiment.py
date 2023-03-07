import argparse
import configparser
import re

import prefect
from prefect import Flow, Parameter, task
from prefect.executors import LocalDaskExecutor
from prefect.run_configs import LocalRun

from scmultiplex.config import (
    compute_workflow_params,
    get_round_names,
    get_workflow_params,
    parse_spacing,
)
from scmultiplex.utils.parse_utils import create_experiment


@task(nout=5)
def create_regexes(well_pattern, raw_ch_pattern, mask_ending, nuc_ending, mem_ending):
    well_regex = re.compile(well_pattern)
    raw_ch_regex = re.compile(raw_ch_pattern)

    mask_regex = re.compile(mask_ending + ".tif")
    nuc_seg_regex = re.compile(nuc_ending + ".tif")
    cell_seg_regex = re.compile(mem_ending + ".tif")

    return well_regex, raw_ch_regex, mask_regex, nuc_seg_regex, cell_seg_regex


@task()
def create_experiment_task(
    name,
    root_dir,
    save_dir,
    overview_spacing,
    spacing,
    fname_barcode_index,
    well_regex,
    raw_ch_regex,
    mask_regex,
    nuc_seg_regex,
    cell_seg_regex,
):
    create_experiment(
        name=name,
        root_dir=root_dir,
        save_dir=save_dir,
        overview_spacing=overview_spacing,
        spacing=spacing,
        fname_barcode_index=fname_barcode_index,
        well_regex=well_regex,
        raw_ch_regex=raw_ch_regex,
        mask_regex=mask_regex,
        nuc_seg_regex=nuc_seg_regex,
        cell_seg_regex=cell_seg_regex,
        logger=prefect.context.get("logger"),
    )


with Flow(
    "Build-Experiment",
    executor=LocalDaskExecutor(),
    run_config=LocalRun(),
) as flow:
    well_pattern = Parameter("well_pattern", default="_[A-Z]{1}[0-9]{2}_")
    raw_ch_pattern = Parameter("raw_ch_pattern", default="C[0-9]{2}O.*_TIF-OVR.tif")
    mask_ending = Parameter("mask_ending", default="MASK")
    nuc_ending = Parameter("nuc_ending", default="NUC_SEG3D_220523")
    mem_ending = Parameter("mem_ending", default="MEM_SEG3D_220523")

    name = Parameter("name", default="20220507GCPLEX_R3")
    root_dir = Parameter(
        "root_dir",
        default="/tungstenfs/scratch/gliberal/Users/repinico/Yokogawa/20220507GCPLEX_R3",
    )
    save_dir = Parameter(
        "save_dir", default="/home/tibuch/Gitrepos/gliberal-scMultipleX/jpynb"
    )
    spacing = Parameter("spacing", default=[0.6, 0.216, 0.216])
    overview_spacing = Parameter("overview_spacing", default=[0.216, 0.216])
    fname_barcode_index = Parameter("fname_barcode_index", default=3)

    regexes = create_regexes(
        well_pattern=well_pattern,
        raw_ch_pattern=raw_ch_pattern,
        mask_ending=mask_ending,
        nuc_ending=nuc_ending,
        mem_ending=mem_ending,
    )
    well_regex, raw_ch_regex, mask_regex, nuc_seg_regex, cell_seg_regex = regexes

    exp = create_experiment_task(
        name=name,
        root_dir=root_dir,
        save_dir=save_dir,
        overview_spacing=overview_spacing,
        spacing=spacing,
        fname_barcode_index=fname_barcode_index,
        well_regex=well_regex,
        raw_ch_regex=raw_ch_regex,
        mask_regex=mask_regex,
        nuc_seg_regex=nuc_seg_regex,
        cell_seg_regex=cell_seg_regex,
    )

def get_config_params(config_file_path):
    round_names = get_round_names(config_file_path)
    config_params = {
        'well_pattern':     ('00BuildExperiment', 'well_pattern'),
        'raw_ch_pattern':   ('00BuildExperiment', 'raw_ch_pattern'),
        'mask_ending':      ('00BuildExperiment', 'mask_ending'),
        'save_dir':         ('00BuildExperiment', 'base_dir_save'),
        }
    common_params = get_workflow_params(config_file_path, config_params)
    
    compute_param = {
        'spacing': (
            parse_spacing,[
                ('00BuildExperiment', 'spacing')
                ]
            ),
        'overview_spacing': (
            parse_spacing,[
                ('00BuildExperiment', 'overview_spacing')
                ]
            ),
        }
    common_params.update(compute_workflow_params(config_file_path, compute_param))
    
    round_params = {}
    for ro in round_names:
        config_params = {
            'name':                 ('00BuildExperiment.round_%s' % ro, 'name'),
            'nuc_ending':           ('00BuildExperiment.round_%s' % ro, 'nuc_ending'),
            'mem_ending':           ('00BuildExperiment.round_%s' % ro, 'mem_ending'),
            'root_dir':             ('00BuildExperiment.round_%s' % ro, 'root_dir'),
            }
        rp = common_params.copy()
        rp.update(get_workflow_params(config_file_path, config_params))
        compute_param = {
            'fname_barcode_index': (
                int,[
                    ('00BuildExperiment.round_%s' % ro, 'fname_barcode_index')
                    ]
                ),
        }
        rp.update(compute_workflow_params(config_file_path, compute_param))
        round_params[ro] = rp
    return round_params

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()

    r_params = get_config_params(args.config)
    for ro, kwargs in r_params.items():
        flow.run(parameters = kwargs)


if __name__ == "__main__":
    main()
