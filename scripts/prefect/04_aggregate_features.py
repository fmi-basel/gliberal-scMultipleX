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
    get_workflow_params,
)
from scmultiplex.utils.accumulate_utils import (
    save_tidy_plate_well_org,
    save_tidy_plate_well_org_nuc,
    save_tidy_plate_well_org_mem
)

from scmultiplex.logging import setup_prefect_handlers
from scmultiplex.utils import get_core_count

@task()
def load_experiment(exp_path):
    exp = Experiment()
    exp.load(exp_path)
    return exp


@task()
def save_tidy_task(exp, org_seg_ch, nuc_seg_ch, mem_seg_ch):

    for tidy_task in [save_tidy_plate_well_org, save_tidy_plate_well_org_nuc, save_tidy_plate_well_org_mem]:
        try:
            tidy_task(exp, (org_seg_ch, nuc_seg_ch, mem_seg_ch))
        except RuntimeWarning as e:
            logger = prefect.context.get("logger")
            logger.info('%s' % str(e))


def run_flow(r_params, cpus):
    with Flow(
        "Tidy Organoid, Nuclear, and Membrane Features",
        executor=LocalDaskExecutor(scheduler="threads", num_workers=cpus),
        run_config=LocalRun(),
    ) as flow:
        exp_path = Parameter("exp_path")
        org_seg_ch = Parameter("org_seg_ch")
        nuc_seg_ch = Parameter("nuc_seg_ch")
        mem_seg_ch = Parameter("mem_seg_ch")

        exps = load_experiment(exp_path)

        save_org = save_tidy_task(exps, org_seg_ch, nuc_seg_ch, mem_seg_ch)

    ret = 0
    for ro, kwargs in r_params.items():
        state = flow.run(parameters=kwargs)
        if state.is_failed():
            ret += 1
            break
    return ret


def get_config_params(config_file_path):
    
    round_names = get_round_names(config_file_path)
    round_params = {}
    for ro in round_names:
        config_params = {
            'org_seg_ch':           ('00BuildExperiment.round_%s' % ro, 'organoid_seg_channel'),
            'nuc_seg_ch':           ('00BuildExperiment.round_%s' % ro, 'nuclear_seg_channel'),
            'mem_seg_ch':           ('00BuildExperiment.round_%s' % ro, 'membrane_seg_channel'),
            }
        rp = get_workflow_params(config_file_path, config_params)

        compute_param = {
            'exp_path': (
                summary_csv_path,[
                    ('00BuildExperiment', 'base_dir_save'),
                    ('00BuildExperiment.round_%s' % ro, 'name')
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

    ret = run_flow(r_params, cpus)
    if ret == 0:
        print('%s completed successfully' % os.path.basename(sys.argv[0]))
    return ret


if __name__ == "__main__":
    sys.exit(main())
