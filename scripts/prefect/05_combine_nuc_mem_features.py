# Copyright (C) 2023 Friedrich Miescher Institute for Biomedical Research

##############################################################################
#                                                                            #
# Author: Nicole Repina              <nicole.repina@fmi.ch>                  #
# Author: Tim-Oliver Buchholz        <tim-oliver.buchholz@fmi.ch>            #
# Author: Enrico Tagliavini          <enrico.tagliavini@fmi.ch>              #
#                                                                            #
##############################################################################

import argparse
import prefect

from faim_hcs.hcs.Experiment import Experiment
from prefect import Flow, Parameter, task
from prefect.executors import LocalDaskExecutor
from prefect.run_configs import LocalRun
from traceback import format_exc

from scmultiplex.config import (
    compute_workflow_params,
    get_round_names,
    summary_csv_path,
)
from scmultiplex.utils.accumulate_utils import (
    write_merged_nuc_membrane_features,
    write_nuc_to_mem_linking,
)

from scmultiplex.utils import get_core_count

@task()
def load_experiment(exp_path):
    exp = Experiment()
    exp.load(exp_path)
    return exp


@task()
def write_nuc_to_mem_linking_task(exp):
    try:
        write_nuc_to_mem_linking(exp)
    except RuntimeWarning as e:
        logger = prefect.context.get("logger")
        logger.info('%s' % str(e))
    else:
        write_merged_nuc_membrane_features(exp)


def run_flow(r_params, cpus):
    with Flow(
        "Accumulate Results",
        executor=LocalDaskExecutor(scheduler="threads", num_workers=cpus),
        run_config=LocalRun(),
    ) as flow:
        exp_path = Parameter("exp_path")

        exp = load_experiment(exp_path)

        save_linking = write_nuc_to_mem_linking_task(exp)

    for ro, kwargs in r_params.items():
        flow.run(parameters = kwargs)
    return


def get_config_params(config_file_path):
    
    round_names = get_round_names(config_file_path)
    round_params = {}
    for ro in round_names:
        compute_param = {
            'exp_path': (
                summary_csv_path,[
                    ('00BuildExperiment', 'base_dir_save'),
                    ('00BuildExperiment.round_%s' % ro, 'name')
                    ]
                ),
        }
        rp = compute_workflow_params(config_file_path, compute_param)
        round_params[ro] = rp
    return round_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--cpus", type=int, default=get_core_count())
    args = parser.parse_args()
    cpus = args.cpus

    r_params = get_config_params(args.config)

    run_flow(r_params, cpus)


if __name__ == "__main__":
    main()
