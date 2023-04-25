# Copyright (C) 2023 Friedrich Miescher Institute for Biomedical Research

##############################################################################
#                                                                            #
# Author: Nicole Repina              <nicole.repina@fmi.ch>                  #
# Author: Tim-Oliver Buchholz        <tim-oliver.buchholz@fmi.ch>            #
# Author: Enrico Tagliavini          <enrico.tagliavini@fmi.ch>              #
#                                                                            #
##############################################################################

import argparse
import configparser
import re
import sys

import prefect
from prefect import Flow, Parameter, task
from prefect.executors import LocalDaskExecutor
from prefect.run_configs import LocalRun

from scmultiplex import version
from scmultiplex.config import (
    compute_workflow_params,
    get_round_names,
    get_workflow_params,
    parse_spacing,
)
from scmultiplex.logging import setup_prefect_handlers
from scmultiplex.utils.parse_utils import create_experiment
from scmultiplex.utils import get_core_count


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


def run_flow(r_params, cpus):
    with Flow(
        "Build-Experiment",
        executor=LocalDaskExecutor(scheduler="processes", num_workers=cpus),
        run_config=LocalRun(),
    ) as flow:
        well_pattern = Parameter("well_pattern")
        raw_ch_pattern = Parameter("raw_ch_pattern")
        mask_ending = Parameter("mask_ending")
        nuc_ending = Parameter("nuc_ending")
        mem_ending = Parameter("mem_ending")

        name = Parameter("name")
        root_dir = Parameter("root_dir")
        save_dir = Parameter("save_dir")
        spacing = Parameter("spacing")
        overview_spacing = Parameter("overview_spacing")
        fname_barcode_index = Parameter("fname_barcode_index")

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

    ret = 0
    for ro, kwargs in r_params.items():
        state = flow.run(parameters=kwargs)
        if state.is_failed():
            ret += 1
            break
    return ret


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
    parser.add_argument("--config", required = True)
    parser.add_argument("--cpus", type=int, default=get_core_count())
    parser.add_argument("--prefect-logfile", required = True)

    args = parser.parse_args()
    cpus = args.cpus
    prefect_logfile = args.prefect_logfile
    
    setup_prefect_handlers(prefect.utilities.logging.get_logger(), prefect_logfile)

    print('Running scMultipleX version %s' % version)

    r_params = get_config_params(args.config)
    return run_flow(r_params, cpus)


if __name__ == "__main__":
    sys.exit(main())
