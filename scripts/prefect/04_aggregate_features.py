import argparse
import prefect

from faim_hcs.hcs.Experiment import Experiment
from prefect import Flow, Parameter, task
from prefect.executors import LocalDaskExecutor
from prefect.run_configs import LocalRun

from scmultiplex.config import (
    compute_workflow_params,
    get_round_names,
    summary_csv_path,
    get_workflow_params,
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
def save_tidy_task(exp, org_seg_ch, nuc_seg_ch, mem_seg_ch):

    for tidy_task in [save_tidy_plate_well_org, save_tidy_plate_well_org_nuc, save_tidy_plate_well_org_mem]:
        try:
            tidy_task(exp, (org_seg_ch, nuc_seg_ch, mem_seg_ch))
        except RuntimeWarning as e:
            logger = prefect.context.get("logger")
            logger.info('%s' % str(e))


with Flow(
    "Tidy Organoid, Nuclear, and Membrane Features",
    executor=LocalDaskExecutor(),
    run_config=LocalRun(),
) as flow:
    exp_path = Parameter("exp_path")
    org_seg_ch = Parameter("org_seg_ch")
    nuc_seg_ch = Parameter("nuc_seg_ch")
    mem_seg_ch = Parameter("mem_seg_ch")

    exps = load_experiment(exp_path)

    save_org = save_tidy_task(exps, org_seg_ch, nuc_seg_ch, mem_seg_ch)


def get_config_params(config_file_path):
    
    round_names = get_round_names(config_file_path)
    round_params = {}
    for ro in round_names:
        config_params = {
            'org_seg_ch':           ('00BuildExperiment.round_%s' % ro, 'organoid_seg_channel'),
            'nuc_seg_ch':           ('00BuildExperiment.round_%s' % ro, 'nuclear_seg_channel'),
            'mem_seg_ch':           ('00BuildExperiment.round_%s' % ro, 'membrane_seg_channel'),
            }
        rp = get_workflow_params(config_file_path, config_params)

        compute_param = {
            'exp_path': (
                summary_csv_path,[
                    ('00BuildExperiment', 'base_dir_save'),
                    ('00BuildExperiment.round_%s' % ro, 'name')
                    ]
                ),
        }
        rp.update(compute_workflow_params(config_file_path, compute_param))
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
