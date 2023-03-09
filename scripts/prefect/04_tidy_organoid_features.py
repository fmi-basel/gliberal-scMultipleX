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
    save_tidy_plate_well_org,
    save_tidy_plate_well_org_nuc,
    save_tidy_plate_well_org_mem
)


@task()
def load_experiment(exp_path):
    exp = Experiment()
    exp.load(exp_path)
    return exp


@task()
def save_tidy_task(exp):
    save_tidy_plate_well_org(exp)
    save_tidy_plate_well_org_nuc(exp)
    save_tidy_plate_well_org_mem(exp)


with Flow(
    "Tidy Organoid, Nuclear, and Membrane Features",
    executor=LocalDaskExecutor(),
    run_config=LocalRun(),
) as flow:
    exp_path = Parameter("exp_path", default="/path/to/exp/summary.csv")

    exps = load_experiment(exp_path)

    save_org = save_tidy_task(exps)


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
