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
import configparser
import os
import prefect
import sys

from scmultiplex.faim_hcs.hcs.Experiment import Experiment
from prefect import Flow, Parameter, task
from prefect.executors import LocalDaskExecutor
from prefect.run_configs import LocalRun

from scmultiplex import version
from scmultiplex.config import (
    compute_workflow_params,
    get_round_names,
    summary_csv_path,
)
from scmultiplex.utils.accumulate_utils import (
    merge_platymatch_linking,
    write_nuclear_linking_over_multiplexing_rounds
)

from scmultiplex.logging import setup_prefect_handlers
from scmultiplex.utils import get_core_count

@task()
def load_experiment(exp_path):
    exp = Experiment()
    exp.load(exp_path)
    return exp

@task()
def merge_scmultiplexed_features_task(exp, round_names, round_summary_csvs):
    merge_platymatch_linking(exp, transform = "affine")
    write_nuclear_linking_over_multiplexing_rounds(round_names, round_summary_csvs, transform = "affine")

def run_flow(r_params, cpus):
    with Flow(
        "Merge Nuclear Linking",
        executor=LocalDaskExecutor(scheduler="threads", num_workers=cpus),
        run_config=LocalRun(),
    ) as flow:
        exp_path = Parameter("exp_path")
        round_names = Parameter("round_names")
        round_summary_csv = Parameter("round_summary_csv")

        exp = load_experiment(exp_path)

        merge_scmultiplexed_features_task(exp, round_names, round_summary_csv)

    ret = 0
    state = flow.run(parameters=r_params)
    if state.is_failed():
        ret += 1
    return ret


def get_config_params(config_file_path):

    round_names = get_round_names(config_file_path)

    if len(round_names) < 2:
        raise RuntimeError('At least two rounds are required to perform organoid linking')

    # initialize dictionary that contains list of summary.csv for each round
    rp = {'round_names': round_names, 'round_summary_csv': []}

    for ro in round_names:
        compute_param = {
            'RX_path': (
                summary_csv_path, [
                    ('00BuildExperiment', 'base_dir_save'),
                    ('00BuildExperiment.round_%s' % ro, 'name')
                ]
            ),
        }
        rscsv = compute_workflow_params(config_file_path, compute_param)['RX_path']
        rp['round_summary_csv'].append(rscsv)

    rp['exp_path'] = rp['round_summary_csv'][0] # select summary_csv of first round (R0)

    return rp


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

    ret = run_flow(r_params, cpus)
    if ret == 0:
        print('%s completed successfully' % os.path.basename(sys.argv[0]))
    return ret

if __name__ == "__main__":
    sys.exit(main())
