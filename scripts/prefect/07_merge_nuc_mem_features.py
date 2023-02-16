import argparse
import configparser

from faim_hcs.hcs.Experiment import Experiment
from prefect import Flow, Parameter, task
from prefect.executors import LocalDaskExecutor
from prefect.run_configs import LocalRun

from scmultiplex.utils.accumulate_utils import (
    write_merged_nuc_membrane_features,
    write_nuc_to_mem_linking,
)


@task()
def load_experiment(exp_csv):
    exp = Experiment()
    exp.load(exp_csv)
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
    exp_csv = Parameter("exp_csv", default="/path/to/exp/summary.csv")

    exp = load_experiment(exp_csv)

    save_linking = write_nuc_to_mem_linking_task(exp)

    save_nuc_mem_linking = write_merged_nuc_membrane_features_task(
        exp, upstream_tasks=[save_linking]
    )


def conf_to_dict(config):
    return {
        "exp_csv": config["DEFAULT"]["exp_csv"],
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
