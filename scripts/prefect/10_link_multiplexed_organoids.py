import argparse
import configparser

from prefect import Flow, Parameter, task
from prefect.executors import LocalDaskExecutor
from prefect.run_configs import LocalRun

from scmultiplex.utils.accumulate_utils import (
    write_organoid_linking_over_multiplexing_rounds,
)


@task()
def write_organoid_linking_over_multiplexing_rounds_task(
    round_names, round_summary_csvs
):
    write_organoid_linking_over_multiplexing_rounds(round_names, round_summary_csvs)


with Flow(
    "Link Multiplexed Organoids",
    executor=LocalDaskExecutor(),
    run_config=LocalRun(),
) as flow:
    round_names = Parameter("round_names", default=["R0", "R1", "R2"])
    round_summary_csv = Parameter(
        "round_summary_csv",
        default=[
            "/tungstenfs/scratch/gliberal/Users/repinico/Microscopy/Analysis/20220604_GCPLEX/220604GCPLEX_R0/summary.csv",
            "/tungstenfs/scratch/gliberal/Users/repinico/Microscopy/Analysis/20220604_GCPLEX/220604GCPLEX_R1/summary.csv",
            "/tungstenfs/scratch/gliberal/Users/repinico/Microscopy/Analysis/20220604_GCPLEX/220604GCPLEX_R2/summary.csv",
        ],
    )

    org_ovr_mpx_rounds = write_organoid_linking_over_multiplexing_rounds_task(
        round_names, round_summary_csv
    )


def conf_to_dict(config):
    return {
        "round_names": config["DEFAULT"]["round_names"].split(","),
        "round_summary_csv": config["DEFAULT"]["round_summary_csv"].split(","),
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
