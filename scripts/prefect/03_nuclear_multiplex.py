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
from typing import List

from faim_hcs.hcs.Experiment import Experiment
from prefect import Flow, Parameter, task, unmapped
from prefect.executors import LocalDaskExecutor
from prefect.run_configs import LocalRun

from scmultiplex.config import (
    commasplit,
    compute_workflow_params,
    get_round_names,
    get_workflow_params,
    parse_spacing,
    summary_csv_path,
    spacing_anisotropy_scalar,
)
from scmultiplex.linking.NucleiLinking import link_nuclei


@task()
def load_experiment(exp_csv):
    exp = Experiment()
    exp.load(exp_csv)
    return exp


@task()
def get_organoids(
    exp: Experiment, exluded_plates: List[str], excluded_wells: List[str]
):
    exp.only_iterate_over_wells(False)
    exp.reset_iterator()

    organoids = []
    for organoid in exp:
        if organoid.well.plate.plate_id not in exluded_plates:
            if organoid.well.well_id not in excluded_wells:
                organoids.append(organoid)

    return organoids


@task()
def get_organoids_task(exp: Experiment, exlude_plates: List[str]):
    return get_organoids(exp, exlude_plates)


@task()
def link_nuclei_task(organoid, ovr_channel, segname, rx_name, RX, z_anisotropy, org_seg_ch, nuc_seg_ch):
    link_nuclei(
        organoid=organoid,
        ovr_channel=ovr_channel,
        segname=segname,
        rx_name=rx_name,
        RX=RX,
        z_anisotropy=z_anisotropy,
        org_seg_ch=org_seg_ch,
        nuc_seg_ch=nuc_seg_ch,
    )


with Flow(
    "Nuclei-Linking", executor=LocalDaskExecutor(), run_config=LocalRun()
) as flow:
    rx_name = Parameter("RX_name")
    r0_csv = Parameter("R0_path")
    rx_csv = Parameter("RX_path")
    excluded_plates = Parameter("excluded_plates")
    excluded_wells = Parameter("excluded_wells")
    nuc_ending = Parameter("nuc_ending")
    ovr_channel = Parameter("ovr_channel")
    spacing = Parameter("spacing")
    org_seg_ch = Parameter("org_seg_ch")
    nuc_seg_ch = Parameter("nuc_seg_ch")

    R0 = load_experiment(r0_csv)
    RX = load_experiment(rx_csv)

    r0_organoids = get_organoids(R0, excluded_plates, excluded_wells)

    link_nuclei_task.map(
        r0_organoids,
        unmapped(ovr_channel),
        unmapped(nuc_ending),
        unmapped(rx_name),
        unmapped(RX),
        unmapped(spacing),
        unmapped(org_seg_ch),
        unmapped(nuc_seg_ch),
    )

def get_config_params(config_file_path):
    
    round_names = get_round_names(config_file_path)
    if len(round_names) < 2:
        raise RuntimeError('At least two rounds are required to perform organoid linking')
    rounds_tobelinked = round_names[1:]
    config_params = {
        'ovr_channel':     ('01FeatureExtraction', 'ovr_channel'),
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
        'spacing': (
            parse_spacing,[
                ('00BuildExperiment', 'spacing')
                ]
            ),
        'R0_path': (
                summary_csv_path,[
                    ('00BuildExperiment', 'base_dir_save'),
                    ('00BuildExperiment.round_%s' % round_names[0], 'name')
                    ]
                ),
        }
    common_params.update(compute_workflow_params(config_file_path, compute_param))

    # use same z-anisotropy as used during feature extraction
    parsed_spacing = common_params['spacing']
    common_params['spacing'] = spacing_anisotropy_scalar(parsed_spacing)
    
    round_tobelinked_params = {}
    for ro in round_names:
        rp = common_params.copy()
        rp['RX_name'] = ro
        config_params = {
            'nuc_ending':           ('00BuildExperiment.round_%s' % ro, 'nuc_ending'),
            'nuc_seg_ch':           ('00BuildExperiment.round_%s' % ro, 'nuclear_seg_channel'),
            'org_seg_ch':           ('00BuildExperiment.round_%s' % ro, 'organoid_seg_channel'),
            }
        rp.update(get_workflow_params(config_file_path, config_params))
        compute_param = {
            'RX_path': (
                summary_csv_path,[
                    ('00BuildExperiment', 'base_dir_save'),
                    ('00BuildExperiment.round_%s' % ro, 'name')
                    ]
                ),
        }
        rp.update(compute_workflow_params(config_file_path, compute_param))

        if ro == round_names[0]:
            R0_nuc_seg_ch = rp['nuc_seg_ch']
        else:
            round_tobelinked_params[ro] = rp

    if len(set([R0_nuc_seg_ch] + [x['nuc_seg_ch'] for x in round_tobelinked_params.values()])) > 1:
        raise NotImplementedError('Multiplexed nuclear linking between different channels is not supported.')

    return round_tobelinked_params

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()

    r_params = get_config_params(args.config)
    for ro, kwargs in r_params.items():
        flow.run(parameters = kwargs)


if __name__ == "__main__":
    main()
