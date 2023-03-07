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
from scmultiplex.utils.accumulate_utils import merge_org_linking


@task()
def load_experiment(exp_path):
    exp = Experiment()
    exp.load(exp_path)
    return exp


@task()
def merge_org_linking_task(exp):
    merge_org_linking(exp)


with Flow(
    "Merge Organoid Linking",
    executor=LocalDaskExecutor(),
    run_config=LocalRun(),
) as flow:
    exp_path = Parameter("exp_path", default="/path/to/exp/summary.csv")

    exp = load_experiment(exp_path)

    save_org_linking = merge_org_linking_task(exp)


def get_config_params(config_file_path):
    
    round_names = get_round_names(config_file_path)
    first_round = round_names[0]
    # this script only need the first round
    compute_param = {
        'exp_path': (
            summary_csv_path,[
                ('00BuildExperiment', 'base_dir_save'),
                ('00BuildExperiment.round_%s' % first_round, 'name')
                ]
            ),
    }
    rp = compute_workflow_params(config_file_path, compute_param)
    return rp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()

    r_params = get_config_params(args.config)
    flow.run(parameters = r_params)


if __name__ == "__main__":
    main()
