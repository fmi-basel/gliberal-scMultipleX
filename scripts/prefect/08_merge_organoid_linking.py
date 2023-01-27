import argparse
import configparser

from faim_hcs.hcs.Experiment import Experiment
from prefect import Flow, Parameter, task
from prefect.executors import LocalDaskExecutor
from prefect.run_configs import LocalRun

from scmultiplex.utils.accumulate_utils import merge_org_linking


@task()
def load_experiment(exp_csv):
    exp = Experiment()
    exp.load(exp_csv)
    return exp


@task()
def merge_org_linking_task(exp):
    merge_org_linking(exp)


with Flow(
    "Merge Organoid Linking",
    executor=LocalDaskExecutor(),
    run_config=LocalRun(),
) as flow:
    exp_csv = Parameter("exp_csv", default="/path/to/exp/summary.csv")

    exp = load_experiment(exp_csv)

    save_org_linking = merge_org_linking_task(exp)


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
