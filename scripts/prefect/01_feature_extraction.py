import argparse
import configparser
from typing import List

import prefect
from faim_hcs.hcs.Experiment import Experiment
from faim_hcs.records.WellRecord import WellRecord
from prefect import Flow, Parameter, task, unmapped
from prefect.executors import LocalDaskExecutor
from prefect.run_configs import LocalRun

from scmultiplex.features.FeatureExtraction import (
    extract_2d_ovr,
    extract_organoid_features,
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
def well_feature_extraction_ovr_task(well: WellRecord, ovr_channel: str, name_ovr: str):
    logger = prefect.context.get("logger")
    ovr_seg_img, ovr_seg_tiles = load_ovr(well, ovr_channel)

    if ovr_seg_img is None and ovr_seg_tiles is None:
        logger.warning(f"ovr_seg does not exists. Skipping {well.well_id}.")
    else:
        touching_labels_lst = flag_touching(ovr_seg_img, ovr_seg_tiles)
        df_ovr = extract_2d_ovr(well, ovr_channel, ovr_seg_img, touching_labels_lst)
        save_to_well(well, name_ovr + ovr_channel, df_ovr)


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
def organoid_feature_extraction_task(
    organoid, nuc_ending: str, mem_ending: str, mask_ending: str
):
    extract_organoid_features(
        organoid=organoid,
        nuc_ending=nuc_ending,
        mem_ending=mem_ending,
        mask_ending=mask_ending,
    )


@task()
def link_nuc_to_membrane_task(
    organoid, ovr_channel, nuc_ending, mask_ending, mem_ending, iop_cutoff
):
    link_nuc_to_membrane(
        organoid=organoid,
        ovr_channel=ovr_channel,
        nuc_ending=nuc_ending,
        mask_ending=mask_ending,
        mem_ending=mem_ending,
        iop_cutoff=iop_cutoff,
    )


# all feature extraction in one flow because writing to the same json file
with Flow(
    "Feature-Extraction",
    executor=LocalDaskExecutor(),
    run_config=LocalRun(),
) as flow:
    exp_path = Parameter("exp_path", default="/path/to/exp")
    excluded_plates = Parameter("excluded_plates", default=[])
    excluded_wells = Parameter("excluded_wells", default=[])
    ovr_channel = Parameter("ovr_channel", default="C01")
    mask_ending = Parameter("mask_ending", default="MASK")
    nuc_ending = Parameter("nuc_ending", default="NUC_SEG3D_220523")
    mem_ending = Parameter("mem_ending", default="MEM_SEG3D_220523")
    name_ovr = Parameter("name_ovr", default="regionprops_ovr_")
    iop_cutoff = Parameter("iop_cutoff", default=0.6)

    exp, wells = load_task(exp_path, excluded_plates, excluded_wells)

    well_feature_extraction_ovr_task.map(
        wells, unmapped(ovr_channel), unmapped(name_ovr)
    )

    organoids = get_organoids(exp, mask_ending, excluded_plates, excluded_wells)

    feat_ext = organoid_feature_extraction_task.map(
        organoids, unmapped(nuc_ending), unmapped(mem_ending), unmapped(mask_ending)
    )

    link_nuc_to_membrane_task.map(
        organoids,
        unmapped(ovr_channel),
        unmapped(nuc_ending),
        unmapped(mask_ending),
        unmapped(mem_ending),
        unmapped(iop_cutoff),
        upstream_tasks=[feat_ext],
    )


def conf_to_dict(config):
    return {
        "exp_path": config["DEFAULT"]["exp_path"],
        "excluded_plates": config["DEFAULT"]["excluded_plates"].split(","),
        "excluded_wells": config["DEFAULT"]["excluded_wells"].split(","),
        "ovr_channel": config["DEFAULT"]["ovr_channel"],
        "mask_ending": config["DEFAULT"]["mask_ending"],
        "nuc_ending": config["DEFAULT"]["nuc_ending"],
        "mem_ending": config["DEFAULT"]["mem_ending"],
        "name_ovr": config["DEFAULT"]["name_ovr"],
        "iop_cutoff": float(config["DEFAULT"]["iop_cutoff"]),
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
