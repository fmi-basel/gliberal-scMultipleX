import argparse
import configparser

from faim_hcs.hcs.Experiment import Experiment
from prefect import Flow, Parameter, task
from prefect.executors import LocalDaskExecutor
from prefect.run_configs import LocalRun

from scmultiplex.config import (
    compute_workflow_params,
    get_round_names,
    summary_csv_path,
)
from scmultiplex.utils.accumulate_utils import (
    write_merged_nuc_membrane_features,
    write_nuc_to_mem_linking,
)


@task()
def load_experiment(exp_path):
    exp = Experiment()
    exp.load(exp_path)
    return exp


@task()
def write_nuc_to_mem_linking_task(exp):
    write_nuc_to_mem_linking(exp)


@task()
def write_merged_nuc_membrane_features_task(exp):
    write_merged_nuc_membrane_features(exp)


with Flow(
    "Accumulate Results",
    executor=LocalDaskExecutor(),
    run_config=LocalRun(),
) as flow:
    exp_path = Parameter("exp_path", default="/path/to/exp/summary.csv")

    exp = load_experiment(exp_path)

    save_linking = write_nuc_to_mem_linking_task(exp)

    save_nuc_mem_linking = write_merged_nuc_membrane_features_task(
        exp, upstream_tasks=[save_linking]
    )


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
    args = parser.parse_args()

    r_params = get_config_params(args.config)
    for ro, kwargs in r_params.items():
        flow.run(parameters = kwargs)


if __name__ == "__main__":
    main()
