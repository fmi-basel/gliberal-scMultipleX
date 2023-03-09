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
    merge_platymatch_linking,
    write_nuclear_linking_over_multiplexing_rounds
)


@task()
def load_experiment(exp_path):
    exp = Experiment()
    exp.load(exp_path)
    return exp


@task()
def merge_platymatch_linking_task(exp):
    merge_platymatch_linking(exp)


@task()
def write_nuclear_linking_over_multiplexing_rounds_task(
    round_names, round_summary_csvs
):
    write_nuclear_linking_over_multiplexing_rounds(round_names, round_summary_csvs)


with Flow(
    "Merge Nuclear Linking",
    executor=LocalDaskExecutor(),
    run_config=LocalRun(),
) as flow:
    exp_path = Parameter("exp_path")
    round_names = Parameter("round_names")
    round_summary_csv = Parameter("round_summary_csv")

    exp = load_experiment(exp_path)

    save_platymatch_linking = merge_platymatch_linking_task(exp)

    nuc_ovr_mpx_rounds = write_nuclear_linking_over_multiplexing_rounds_task(
        round_names,
        round_summary_csv,
    )


def get_config_params(config_file_path):

    round_names = get_round_names(config_file_path)

    if len(round_names) < 2:
        raise RuntimeError('At least two rounds are required to perform organoid linking')

    # initialize dictionary that contains list of summary.csv for each round
    rp = {'round_names': round_names, 'round_summary_csv': []}

    for ro in round_names:
        compute_param = {
            'RX_path': (
                summary_csv_path, [
                    ('00BuildExperiment', 'base_dir_save'),
                    ('00BuildExperiment.round_%s' % ro, 'name')
                ]
            ),
        }
        rscsv = compute_workflow_params(config_file_path, compute_param)['RX_path']
        rp['round_summary_csv'].append(rscsv)

    rp['exp_path'] = rp['round_summary_csv'][0] # select summary_csv of first round (R0)

    return rp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()

    r_params = get_config_params(args.config)
    flow.run(parameters = r_params)


if __name__ == "__main__":
    main()
