import argparse
import configparser

from prefect import Flow, Parameter, task
from prefect.executors import LocalDaskExecutor
from prefect.run_configs import LocalRun

from scmultiplex.config import (
    compute_workflow_params,
    get_round_names,
    summary_csv_path,
)
from scmultiplex.utils.accumulate_utils import (
    write_nuclear_linking_over_multiplexing_rounds,
)


@task()
def write_nuclear_linking_over_multiplexing_rounds_task(
    round_names, round_summary_csvs
):
    write_nuclear_linking_over_multiplexing_rounds(round_names, round_summary_csvs)


with Flow(
    "Accumulate Results",
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

    nuc_ovr_mpx_rounds = write_nuclear_linking_over_multiplexing_rounds_task(
        round_names,
        round_summary_csv,
    )


def get_config_params(config_file_path):
    
    round_names = get_round_names(config_file_path)
    rp = {'round_names': round_names, 'round_summary_csv': []}
    if len(round_names) < 2:
        raise RuntimeError('At least two rounds are required to perform organoid linking')
   
    for ro in round_names:
        compute_param = {
            'RX_path': (
                summary_csv_path,[
                    ('00BuildExperiment', 'base_dir_save'),
                    ('00BuildExperiment.round_%s' % ro, 'name')
                    ]
                ),
        }
        rscsv = compute_workflow_params(config_file_path, compute_param)['RX_path']
        rp['round_summary_csv'].append(rscsv)
    return rp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()

    r_params = get_config_params(args.config)
    flow.run(parameters = r_params)


if __name__ == "__main__":
    main()
