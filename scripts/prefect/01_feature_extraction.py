# Copyright (C) 2023 Friedrich Miescher Institute for Biomedical Research

##############################################################################
#                                                                            #
# Author: Nicole Repina              <nicole.repina@fmi.ch>                  #
# Author: Tim-Oliver Buchholz        <tim-oliver.buchholz@fmi.ch>            #
# Author: Enrico Tagliavini          <enrico.tagliavini@fmi.ch>              #
#                                                                            #
##############################################################################

import argparse
import configparser
import sys
from typing import List

import prefect
from scmultiplex.faim_hcs.hcs.Experiment import Experiment
from scmultiplex.faim_hcs.records.WellRecord import WellRecord
from prefect import Flow, Parameter, task, unmapped
from prefect.executors import LocalDaskExecutor
from prefect.run_configs import LocalRun

import scmultiplex.config
from scmultiplex import version
from scmultiplex.features.FeatureFunctions import set_spacing
from scmultiplex.logging import setup_prefect_handlers
from scmultiplex.utils import get_core_count

from scmultiplex.config import (
    commasplit,
    compute_workflow_params,
    get_round_names,
    get_workflow_params,
    parse_spacing,
    summary_csv_path,
    spacing_anisotropy_tuple,
    str2bool
)
from scmultiplex.features.FeatureExtraction import (
    extract_organoid_features,
    extract_well_features,
    link_nuc_to_membrane,
)
from scmultiplex.features.FeatureFunctions import flag_touching
from scmultiplex.utils.exclude_utils import exclude_conditions

# wrap a function in a prefect task
from scmultiplex.utils.load_utils import load_experiment, load_ovr
from scmultiplex.utils.save_utils import save_to_well


@task(nout=2)
def load_task(exp_path: str, excluded_plates: List[str], excluded_wells: List[str]):
    exp = load_experiment(exp_path)
    wells = exclude_conditions(exp, excluded_plates, excluded_wells)
    return exp, wells


# def exclude_conditions_task(exp: Experiment, excluded_plates: List[str], excluded_wells: List[str]):
#     return exclude_conditions_task(exp, excluded_plates, excluded_wells)


@task()
def well_feature_extraction_ovr_task(well: WellRecord, org_seg_ch: str):
    extract_well_features(
        well=well,
        ovr_channel=org_seg_ch,
    )
    return


@task()
def get_organoids(
    exp: Experiment,
    mask_ending: str,
    excluded_plates: List[str],
    excluded_wells: List[str],
):
    exp.only_iterate_over_wells(False)
    exp.reset_iterator()
    organoids = []
    for organoid in exp:
        if mask_ending in organoid.segmentations.keys() is None:
            continue  # skip organoids that don't have a mask (this will never happen)

        if organoid.well.plate.plate_id in excluded_plates:
            continue  # skip these timepoints

        if organoid.well.well_id in excluded_wells:
            continue  # skip these wells

        organoids.append(organoid)
    return organoids


@task()
def organoid_feature_extraction_and_linking_task(
    organoid, nuc_ending: str, mem_ending: str, mask_ending: str, spacing: List[float], measure_morphology,
        org_seg_ch, nuc_seg_ch, mem_seg_ch, iop_cutoff):

    set_spacing(spacing)
    extract_organoid_features(
        organoid=organoid,
        nuc_ending=nuc_ending,
        mem_ending=mem_ending,
        mask_ending=mask_ending,
        spacing=tuple(spacing),
        measure_morphology=measure_morphology,
        organoid_seg_channel=org_seg_ch,
        nuclear_seg_channel=nuc_seg_ch,
        membrane_seg_channel=mem_seg_ch,
    )
    link_nuc_to_membrane(
        organoid=organoid,
        ovr_channel=org_seg_ch,
        nuc_ending=nuc_ending,
        mask_ending=mask_ending,
        mem_ending=mem_ending,
        iop_cutoff=iop_cutoff,
    )
    return


# all feature extraction in one flow because writing to the same json file
def run_flow(r_params, cpus):
    with Flow(
        "Feature-Extraction",
        executor=LocalDaskExecutor(scheduler="processes", num_workers=cpus),
        run_config=LocalRun(),
    ) as flow:
        exp_path = Parameter("exp_path")
        excluded_plates = Parameter("excluded_plates")
        excluded_wells = Parameter("excluded_wells")
        mask_ending = Parameter("mask_ending")
        nuc_ending = Parameter("nuc_ending")
        mem_ending = Parameter("mem_ending")
        iop_cutoff = Parameter("iop_cutoff")
        spacing = Parameter("spacing")
        measure_morphology = Parameter("measure_morphology")
        org_seg_ch = Parameter("org_seg_ch")
        nuc_seg_ch = Parameter("nuc_seg_ch")
        mem_seg_ch = Parameter("mem_seg_ch")

        exp, wells = load_task(exp_path, excluded_plates, excluded_wells)

        wfeo_t = well_feature_extraction_ovr_task.map(
            wells, unmapped(org_seg_ch)
        )

        organoids = get_organoids(exp, mask_ending, excluded_plates, excluded_wells, upstream_tasks = [wfeo_t])
        
        organoid_feature_extraction_and_linking_task.map(
            organoids,
            unmapped(nuc_ending),
            unmapped(mem_ending),
            unmapped(mask_ending),
            unmapped(spacing),
            unmapped(measure_morphology),
            unmapped(org_seg_ch),
            unmapped(nuc_seg_ch),
            unmapped(mem_seg_ch),
            unmapped(iop_cutoff),
            upstream_tasks = [organoids],
        )

    ret = 0
    for ro, kwargs in r_params.items():
        state = flow.run(parameters=kwargs)
        if state.is_failed():
            ret += 1
            break
    return ret


def get_config_params(config_file_path):
    
    round_names = get_round_names(config_file_path)
    config_params = {
        'mask_ending':     ('00BuildExperiment', 'mask_ending'),
        'measure_morphology': ('01FeatureExtraction', 'measure_morphology'),
        }
    common_params = get_workflow_params(config_file_path, config_params)

    compute_param = {
        'excluded_plates': (
            commasplit,[
                ('01FeatureExtraction', 'excluded_plates')
                ]
            ),
        'excluded_wells': (
            commasplit,[
                ('01FeatureExtraction', 'excluded_wells')
                ]
            ),
        'iop_cutoff': (
            float,[
                ('01FeatureExtraction', 'iop_cutoff')
                ]
            ),
        'measure_morphology': (
            str2bool,[
                ('01FeatureExtraction', 'measure_morphology')
            ]
        ),
        'spacing': (
            parse_spacing,[
                ('00BuildExperiment', 'spacing')
                ]
            ),

        }
    common_params.update(compute_workflow_params(config_file_path, compute_param))

    # for feature extraction, use spacing normalized to x-dim spacing
    parsed_spacing = common_params['spacing']
    common_params['spacing'] = spacing_anisotropy_tuple(parsed_spacing)
    
    round_params = {}
    for ro in round_names:
        config_params = {
            'nuc_ending':           ('00BuildExperiment.round_%s' % ro, 'nuc_ending'),
            'mem_ending':           ('00BuildExperiment.round_%s' % ro, 'mem_ending'),
            'org_seg_ch':           ('00BuildExperiment.round_%s' % ro, 'organoid_seg_channel'),
            'nuc_seg_ch':           ('00BuildExperiment.round_%s' % ro, 'nuclear_seg_channel'),
            'mem_seg_ch':           ('00BuildExperiment.round_%s' % ro, 'membrane_seg_channel'),
            }
        rp = common_params.copy()
        rp.update(get_workflow_params(config_file_path, config_params))
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
    parser.add_argument("--config", required = True)
    parser.add_argument("--cpus", type=int, default=get_core_count())
    parser.add_argument("--prefect-logfile", required = True)

    args = parser.parse_args()
    cpus = args.cpus
    prefect_logfile = args.prefect_logfile

    setup_prefect_handlers(prefect.utilities.logging.get_logger(), prefect_logfile)

    print('Running scMultipleX version %s' % version)

    r_params = get_config_params(args.config)

    return run_flow(r_params, cpus)


if __name__ == "__main__":
    sys.exit(main())
