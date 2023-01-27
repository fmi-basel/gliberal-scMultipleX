import argparse
import configparser

from faim_hcs.hcs.Experiment import Experiment
from prefect import Flow, Parameter, task
from prefect.executors import LocalDaskExecutor
from prefect.run_configs import LocalRun

from scmultiplex.utils.accumulate_utils import save_tidy_plate_well_org


@task()
def load_experiment(exp_csv):
    exp = Experiment()
    exp.load(exp_csv)
    return exp


@task()
def save_tidy_plate_well_org_task(exp):
    save_tidy_plate_well_org(exp)


with Flow(
    "Tidy Organoid Features",
    executor=LocalDaskExecutor(),
    run_config=LocalRun(),
) as flow:
    exp_csv = Parameter("exp_csv", default="/path/to/exp/summary.csv")

    exps = load_experiment(exp_csv)

    save_org = save_tidy_plate_well_org_task(exps)


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
