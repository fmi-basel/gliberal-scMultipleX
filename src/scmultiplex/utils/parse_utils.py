# Copyright (C) 2023 Friedrich Miescher Institute for Biomedical Research

##############################################################################
#                                                                            #
# Author: Nicole Repina              <nicole.repina@fmi.ch>                  #
# Author: Tim-Oliver Buchholz        <tim-oliver.buchholz@fmi.ch>            #
# Author: Enrico Tagliavini          <enrico.tagliavini@fmi.ch>              #
#                                                                            #
##############################################################################

import logging
from glob import glob
from os.path import basename, exists, isdir, join, split, splitext
from typing import List, Pattern, Tuple

from tqdm import tqdm

from scmultiplex.faim_hcs.hcs.Experiment import Experiment
from scmultiplex.faim_hcs.records.OrganoidRecord import OrganoidRecord
from scmultiplex.faim_hcs.records.PlateRecord import PlateRecord
from scmultiplex.faim_hcs.records.WellRecord import WellRecord


def prepare_and_add_organoids(
    organoid_parent_dir: str,
    well: WellRecord,
    raw_ch_regex: Pattern,
    mask_regex: Pattern,
    nuc_seg_regex: Pattern,
    cell_seg_regex: Pattern,
    spacing: Tuple[float],
    logger=logging.getLogger("HCS_Experiment"),
):
    organoids = glob(join(organoid_parent_dir, well.well_id, "*"))
    for organoid in tqdm(organoids, leave=False):
        add_organoid(
            well=well,
            organoid_path=organoid,
            raw_ch_regex=raw_ch_regex,
            mask_regex=mask_regex,
            nuc_seg_regex=nuc_seg_regex,
            cell_seg_regex=cell_seg_regex,
            spacing=spacing,
            logger=logger,
        )


def prepare_and_add_well(
    plate: PlateRecord,
    well_id: str,
    ovr_mips: List[str],
    overview_spacing: Tuple[float],
    well_regex: Pattern,
    mip_ovr_name,
    org_seg_name,
    raw_file_to_raw_name=lambda p: splitext(p)[0][-3:],
    seg_file_to_seg_name=lambda p: splitext(p)[0][-3:],
    well_path_to_well_id=lambda p: basename(p).split("_")[3],
):
    raw_files = get_well_overview_mips(
        overview_mips=ovr_mips, well_id=well_id, well_regex=well_regex
    )

    seg_files = get_well_overview_segs(
        plate=plate,
        well_id=well_id,
        well_path_to_well_id=well_path_to_well_id,
        mip_ovr_name=mip_ovr_name,
        org_seg_name=org_seg_name,
    )
    return add_well(
        plate=plate,
        well_id=well_id,
        raw_files=raw_files,
        seg_files=seg_files,
        overview_spacing=overview_spacing,
        raw_file_to_raw_name=raw_file_to_raw_name,
        seg_file_to_seg_name=seg_file_to_seg_name,
    )


def add_well(
    plate: PlateRecord,
    well_id: str,
    raw_files: List[str],
    seg_files: List[str],
    overview_spacing: Tuple[float],
    raw_file_to_raw_name=lambda p: splitext(p)[0][-3:],
    seg_file_to_seg_name=lambda p: splitext(p)[0][-3:],
):
    w = WellRecord(plate=plate, well_id=well_id, save_dir=plate.plate_dir)

    for raw_file in raw_files:
        raw_name = raw_file_to_raw_name(raw_file)
        w.add_raw_data(raw_name, raw_file, spacing=overview_spacing)

    for seg_file in seg_files:
        seg_name = seg_file_to_seg_name(seg_file)
        w.add_segmentation(seg_name, seg_file)

    return w


def add_organoid(
    well: WellRecord,
    organoid_path: str,
    raw_ch_regex: Pattern,
    mask_regex: Pattern,
    nuc_seg_regex: Pattern,
    cell_seg_regex: Pattern,
    spacing: Tuple[float],
    logger=logging.getLogger("HCS_Experiment"),
):
    org = OrganoidRecord(well, basename(organoid_path), save_dir=well.well_dir)
    organoid_files = glob(join(organoid_path, "*.tif"))
    for f in organoid_files:
        raw_channel = raw_ch_regex.findall(f)
        mask = mask_regex.findall(f)
        nuc_seg = nuc_seg_regex.findall(f)
        cell_seg = cell_seg_regex.findall(f)
        if len(raw_channel) > 0:
            org.add_raw_data(raw_channel[0][:3], f, spacing=spacing)
        elif len(mask) > 0:
            org.add_segmentation(mask[0][:-4], f)
        elif len(nuc_seg) > 0:
            org.add_segmentation(nuc_seg[0][:-4], f)
        elif len(cell_seg) > 0:
            org.add_segmentation(cell_seg[0][:-4], f)
        else:
            logger.warning(f"Unknown file: {f}")


def get_well_overview_mips(overview_mips: List[str], well_id: str, well_regex: Pattern):
    return list(
        filter(
            lambda p: well_regex.findall(basename(p))[0][1:-1] == well_id, overview_mips
        )
    )


def get_well_overview_segs(
    plate: PlateRecord,
    well_id: str,
    well_path_to_well_id: lambda p: basename(p).split("_")[3],
    mip_ovr_name,
    org_seg_name,
):
    seg_mips = glob(
        join(
            plate.experiment.root_dir,
            plate.plate_id,
            mip_ovr_name + "_SEG",
            org_seg_name,
            "*.tif",
        )
    )
    return list(filter(lambda p: well_path_to_well_id(p) == well_id, seg_mips))


def create_experiment(
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
    mip_ovr_name,
    org_seg_name,
    logger=logging.getLogger("HCS_Experiment"),
):
    def raw_file_to_raw_name(p):
        return splitext(p)[0][-3:]

    def seg_file_to_seg_name(p):
        return splitext(p)[0][-3:]

    def well_path_to_well_id(p):
        return basename(p).split("_")[fname_barcode_index]

    exp = Experiment(
        name=name,
        root_dir=root_dir,
        save_dir=save_dir,
    )

    plates = glob(exp.root_dir + "/*")
    for p in plates:
        if isdir(p):
            plate = PlateRecord(
                experiment=exp, plate_id=split(p)[1], save_dir=exp.get_experiment_dir()
            )

            ovr_mips = glob(join(exp.root_dir, plate.plate_id, mip_ovr_name, "*.tif"))

            well_ids = [well_regex.findall(basename(om))[0][1:-1] for om in ovr_mips]

            wells = []
            for well_id in well_ids:
                logger.info(
                    f"Add well {well_id} to plate {plate.plate_id} "
                    f"of experiment {exp.name}."
                )
                wells.append(
                    prepare_and_add_well(
                        plate=plate,
                        well_id=well_id,
                        ovr_mips=ovr_mips,
                        overview_spacing=overview_spacing,
                        well_regex=well_regex,
                        mip_ovr_name=mip_ovr_name,
                        org_seg_name=org_seg_name,
                        raw_file_to_raw_name=raw_file_to_raw_name,
                        seg_file_to_seg_name=seg_file_to_seg_name,
                        well_path_to_well_id=well_path_to_well_id,
                    )
                )

            organoid_parent_dir = join(
                exp.root_dir, plate.plate_id, org_seg_name + "_ROI"
            )
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
                        logger=logger,
                    )

    exp.save()
