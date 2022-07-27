import argparse
import configparser
import re

import prefect
from prefect import Flow, Parameter, task

from scmultiplex.utils.parse_utils import create_experiment


def conf_to_dict(config):
    return {
        "well_pattern": config["DEFAULT"]["well_pattern"],
        "raw_ch_pattern": config["DEFAULT"]["raw_ch_pattern"],
        "mask_ending": config["DEFAULT"]["mask_ending"],
        "nuc_ending": config["DEFAULT"]["nuc_ending"],
        "mem_ending": config["DEFAULT"]["mem_ending"],
        "name": config["EXP"]["name"],
        "root_dir": config["EXP"]["root_dir"],
        "save_dir": config["EXP"]["save_dir"],
        "spacing": tuple(float(v) for v in config["EXP"]["spacing"].split(",")),
        "overview_spacing": tuple(
            float(v) for v in config["EXP"]["spacing"].split(",")
        ),
        "fname_barcode_index": int(config["EXP"]["fname_barcode_index"]),
    }


@task(nout=5)
def create_regexes(well_pattern, raw_ch_pattern, mask_ending, nuc_ending, mem_ending):
    well_regex = re.compile(well_pattern)
    raw_ch_regex = re.compile(raw_ch_pattern)

    mask_regex = re.compile(mask_ending + ".tif")
    nuc_seg_regex = re.compile(nuc_ending + ".tif")
    cell_seg_regex = re.compile(mem_ending + ".tif")

    return well_regex, raw_ch_regex, mask_regex, nuc_seg_regex, cell_seg_regex


@task()
def create_experiment_task(
    name,
    root_dir,
    save_dir,
    overview_spacing,
    spacing,
    fname_barcode_index,
    well_regex,
    raw_ch_regex,
    mask_regex,
    nuc_seg_regex,
    cell_seg_regex,
):
    create_experiment(
        name=name,
        root_dir=root_dir,
        save_dir=save_dir,
        overview_spacing=overview_spacing,
        spacing=spacing,
        fname_barcode_index=fname_barcode_index,
        well_regex=well_regex,
        raw_ch_regex=raw_ch_regex,
        mask_regex=mask_regex,
        nuc_seg_regex=nuc_seg_regex,
        cell_seg_regex=cell_seg_regex,
        logger=prefect.context.get("logger"),
    )


with Flow("Build-Experiment") as flow:
    well_pattern = Parameter("well_pattern", default="_[A-Z]{1}[0-9]{2}_")
    raw_ch_pattern = Parameter("raw_ch_pattern", default="C[0-9]{2}O.*_TIF-OVR.tif")
    mask_ending = Parameter("mask_ending", default="MASK")
    nuc_ending = Parameter("nuc_ending", default="NUC_SEG3D_220523")
    mem_ending = Parameter("mem_ending", default="MEM_SEG3D_220523")

    name = Parameter("name", default="20220507GCPLEX_R3")
    root_dir = Parameter(
        "root_dir",
        default="/tungstenfs/scratch/gliberal/Users/repinico/Yokogawa/20220507GCPLEX_R3",
    )
    save_dir = Parameter(
        "save_dir", default="/home/tibuch/Gitrepos/gliberal-scMultipleX/jpynb"
    )
    spacing = Parameter("spacing", default=[0.6, 0.216, 0.216])
    overview_spacing = Parameter("overview_spacing", default=[0.216, 0.216])
    fname_barcode_index = Parameter("fname_barcode_index", default=3)

    regexes = create_regexes(
        well_pattern=well_pattern,
        raw_ch_pattern=raw_ch_pattern,
        mask_ending=mask_ending,
        nuc_ending=nuc_ending,
        mem_ending=mem_ending,
    )
    well_regex, raw_ch_regex, mask_regex, nuc_seg_regex, cell_seg_regex = regexes

    exp = create_experiment(
        name=name,
        root_dir=root_dir,
        save_dir=save_dir,
        overview_spacing=overview_spacing,
        spacing=spacing,
        fname_barcode_index=fname_barcode_index,
        well_regex=well_regex,
        raw_ch_regex=raw_ch_regex,
        mask_regex=mask_regex,
        nuc_seg_regex=nuc_seg_regex,
        cell_seg_regex=cell_seg_regex,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    kwargs = conf_to_dict(config)

    flow.run(parameters=kwargs)


if __name__ == "__main__":
    main()
