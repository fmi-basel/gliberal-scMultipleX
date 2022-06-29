import argparse
import configparser
import re
from glob import glob
from os.path import splitext, basename, isdir, split, join, exists

from faim_hcs.hcs.Experiment import Experiment
from faim_hcs.records.PlateRecord import PlateRecord
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    kwargs = conf_to_dict(config)

    well_regex = re.compile(kwargs["well_pattern"])
    raw_ch_regex = re.compile(kwargs["raw_ch_pattern"])
    mask_ending = kwargs["mask_ending"]
    nuc_ending = kwargs["nuc_ending"]
    mem_ending = kwargs["mem_ending"]

    # nuc_seg_regex = re.compile("_NUC_SEG.*.tif") #add here the neural network name
    mask_regex = re.compile(mask_ending + ".tif")
    nuc_seg_regex = re.compile(nuc_ending + ".tif")
    cell_seg_regex = re.compile(mem_ending + ".tif")

    raw_file_to_raw_name = lambda p: splitext(p)[0][-3:]
    seg_file_to_seg_name = lambda p: splitext(p)[0][-3:]
    well_path_to_well_id = lambda p: basename(p).split('_')[kwargs["fname_barcode_index"]]

    exp = Experiment(
        name=kwargs["name"],
        root_dir=kwargs["root_dir"],
        save_dir=kwargs["save_dir"],
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
                wells.append(prepare_and_add_well(plate=plate,
                                                  well_id=well_id,
                                                  ovr_mips=ovr_mips,
                                                  overview_spacing=kwargs[
                                                      "overview_spacing"],
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
                        spacing=kwargs["spacing"])

    exp.save()


if __name__ == "__main__":
    main()