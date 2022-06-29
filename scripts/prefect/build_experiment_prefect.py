import argparse
import configparser
import re
from glob import glob
from os.path import splitext, basename, isdir, split, join, exists

import prefect
from faim_hcs.hcs.Experiment import Experiment
from faim_hcs.records.PlateRecord import PlateRecord
from prefect import Flow, Parameter, task
from prefect.executors import LocalDaskExecutor
from scmultiplex.utils.parse_utils import prepare_and_add_well, \
    prepare_and_add_organoids


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
        "spacing": tuple([float(v) for v in config["EXP"]["spacing"].split(
            ',')]),
        "overview_spacing": tuple([float(v) for v in config["EXP"][
            "spacing"].split(
            ',')]),
        "fname_barcode_index": int(config["EXP"]["fname_barcode_index"])
    }


@task(nout=5)
def create_regexes(well_pattern,
                   raw_ch_pattern,
                   mask_ending,
                   nuc_ending,
                   mem_ending):
    well_regex = re.compile(well_pattern)
    raw_ch_regex = re.compile(raw_ch_pattern)

    mask_regex = re.compile(mask_ending + ".tif")
    nuc_seg_regex = re.compile(nuc_ending + ".tif")
    cell_seg_regex = re.compile(mem_ending + ".tif")

    return well_regex, raw_ch_regex, mask_regex, nuc_seg_regex, cell_seg_regex


@task()
def create_experiment(name, root_dir,
                      save_dir,
                      overview_spacing,
                      spacing,
                      fname_barcode_index,
                      well_regex,
                      raw_ch_regex,
                      mask_regex,
                      nuc_seg_regex,
                      cell_seg_regex
                      ):
    logger = prefect.context.get("logger")

    raw_file_to_raw_name = lambda p: splitext(p)[0][-3:]
    seg_file_to_seg_name = lambda p: splitext(p)[0][-3:]
    well_path_to_well_id = lambda p: basename(p).split('_')[
        fname_barcode_index]

    exp = Experiment(
        name=name,
        root_dir=root_dir,
        save_dir=save_dir,
    )

    plates = glob(exp.root_dir + "/*")
    for p in plates:
        if isdir(p):
            plate = PlateRecord(
                experiment=exp, plate_id=split(p)[1],
                save_dir=exp.get_experiment_dir()
            )

            ovr_mips = glob(
                join(exp.root_dir, plate.plate_id, "TIF_OVR_MIP", "*.tif"))

            well_ids = [well_regex.findall(basename(om))[0][1:-1] for om in
                        ovr_mips]

            wells = []
            for well_id in well_ids:
                logger.info(f"Add well {well_id} to plate {plate.plate_id} "
                            f"of experiment {exp.name}.")
                wells.append(prepare_and_add_well(plate=plate,
                                                  well_id=well_id,
                                                  ovr_mips=ovr_mips,
                                                  overview_spacing=overview_spacing,
                                                  well_regex=well_regex,
                                                  raw_file_to_raw_name=raw_file_to_raw_name,
                                                  seg_file_to_seg_name=seg_file_to_seg_name,
                                                  well_path_to_well_id=well_path_to_well_id
                                                  ))

            organoid_parent_dir = join(exp.root_dir, plate.plate_id,
                                       "obj_v0.3_ROI")
            if exists(organoid_parent_dir):
                for well in wells:
                    prepare_and_add_organoids(
                        organoid_parent_dir=organoid_parent_dir,
                        well=well,
                        raw_ch_regex=raw_ch_regex,
                        mask_regex=mask_regex,
                        nuc_seg_regex=nuc_seg_regex,
                        cell_seg_regex=cell_seg_regex,
                        spacing=spacing,
                        logger=logger)

    exp.save()


with Flow("Build-Experiment") as flow:
    well_pattern = Parameter("well_pattern", default="_[A-Z]{1}[0-9]{2}_")
    raw_ch_pattern = Parameter("raw_ch_pattern",
                               default="C[0-9]{2}O.*_TIF-OVR.tif")
    mask_ending = Parameter("mask_ending", default="MASK")
    nuc_ending = Parameter("nuc_ending", default="NUC_SEG3D_220523")
    mem_ending = Parameter("mem_ending", default="MEM_SEG3D_220523")

    name = Parameter("name", default="20220507GCPLEX_R3")
    root_dir = Parameter("root_dir",
                         default="/tungstenfs/scratch/gliberal/Users/repinico/Yokogawa/20220507GCPLEX_R3")
    save_dir = Parameter("save_dir",
                         default="/home/tibuch/Gitrepos/gliberal-scMultipleX/jpynb")
    spacing = Parameter("spacing", default=[0.6, 0.216, 0.216])
    overview_spacing = Parameter("overview_spacing", default=[0.216, 0.216])
    fname_barcode_index = Parameter("fname_barcode_index", default=3)

    regexes = create_regexes(well_pattern=well_pattern,
                             raw_ch_pattern=raw_ch_pattern,
                             mask_ending=mask_ending,
                             nuc_ending=nuc_ending,
                             mem_ending=mem_ending)
    well_regex, raw_ch_regex, mask_regex, nuc_seg_regex, cell_seg_regex = regexes

    exp = create_experiment(name=name,
                            root_dir=root_dir,
                            save_dir=save_dir,
                            overview_spacing=overview_spacing,
                            spacing=spacing,
                            fname_barcode_index=fname_barcode_index,
                            well_regex=well_regex,
                            raw_ch_regex=raw_ch_regex,
                            mask_regex=mask_regex,
                            nuc_seg_regex=nuc_seg_regex,
                            cell_seg_regex=cell_seg_regex
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
