import logging
from glob import glob
from os.path import isdir, split, basename, join, splitext, exists
from typing import List, Pattern, Tuple

from faim_hcs.hcs.Experiment import Experiment
from faim_hcs.records.OrganoidRecord import OrganoidRecord
from faim_hcs.records.PlateRecord import PlateRecord
from faim_hcs.records.WellRecord import WellRecord
from tqdm import tqdm


def prepare_and_add_organoids(organoid_parent_dir: str,
                              well: WellRecord,
                              raw_ch_regex: Pattern,
                              mask_regex: Pattern,
                              nuc_seg_regex: Pattern,
                              cell_seg_regex: Pattern,
                              spacing: Tuple[float],
                              logger=logging):
    organoids = glob(join(organoid_parent_dir, well.well_id, "*"))
    for organoid in tqdm(organoids, leave=False):
        add_organoid(well=well,
                     organoid_path=organoid,
                     raw_ch_regex=raw_ch_regex,
                     mask_regex=mask_regex,
                     nuc_seg_regex=nuc_seg_regex,
                     cell_seg_regex=cell_seg_regex,
                     spacing=spacing,
                     logger=logger)


def prepare_and_add_well(plate: PlateRecord,
                         well_id: str,
                         ovr_mips: List[str],
                         overview_spacing: Tuple[float],
                         well_regex: Pattern,
                         raw_file_to_raw_name=lambda p: splitext(p)[0][-3:],
                         seg_file_to_seg_name=lambda p: splitext(p)[0][-3:],
                         well_path_to_well_id=lambda p: basename(p).split(
                             '_')[3]):
    raw_files = get_well_overview_mips(overview_mips=ovr_mips,
                                       well_id=well_id,
                                       well_regex=well_regex)

    seg_files = get_well_overview_segs(plate=plate,
                                       well_id=well_id,
                                       well_path_to_well_id=well_path_to_well_id
                                       )
    return add_well(plate=plate,
             well_id=well_id,
             raw_files=raw_files,
             seg_files=seg_files,
             overview_spacing=overview_spacing,
             raw_file_to_raw_name=raw_file_to_raw_name,
             seg_file_to_seg_name=seg_file_to_seg_name)


def add_well(plate: PlateRecord, well_id: str,
             raw_files: List[str],
             seg_files: List[str],
             overview_spacing: Tuple[float],
             raw_file_to_raw_name=lambda p: splitext(p)[0][-3:],
             seg_file_to_seg_name=lambda p: splitext(p)[0][-3:]):
    w = WellRecord(plate=plate, well_id=well_id, save_dir=plate.plate_dir)

    for raw_file in raw_files:
        raw_name = raw_file_to_raw_name(raw_file)
        w.add_raw_data(raw_name, raw_file, spacing=overview_spacing)

    for seg_file in seg_files:
        seg_name = seg_file_to_seg_name(seg_file)
        w.add_segmentation(seg_name, seg_file)

    return w


def add_organoid(well: WellRecord,
                 organoid_path: str,
                 raw_ch_regex: Pattern,
                 mask_regex: Pattern,
                 nuc_seg_regex: Pattern,
                 cell_seg_regex: Pattern,
                 spacing: Tuple[float],
                 logger=logging):
    org = OrganoidRecord(well, basename(organoid_path), save_dir=well.well_dir)
    organoid_files = glob(join(organoid_path, "*.tif"))
    for f in organoid_files:
        raw_channel = raw_ch_regex.findall(f)
        mask = mask_regex.findall(f)
        nuc_seg = nuc_seg_regex.findall(f)
        cell_seg = cell_seg_regex.findall(f)
        if len(raw_channel) > 0:
            org.add_raw_data(
                raw_channel[0][:3], f, spacing=spacing
            )
        elif len(mask) > 0:
            org.add_segmentation(mask[0][:-4], f)
        elif len(nuc_seg) > 0:
            org.add_segmentation(nuc_seg[0][:-4], f)
        elif len(cell_seg) > 0:
            org.add_segmentation(cell_seg[0][:-4], f)
        else:
            logger.warning(f"Unknown file: {f}")


def get_well_overview_mips(overview_mips: List[str], well_id: str,
                           well_regex: Pattern):
    return list(
        filter(
            lambda p: well_regex.findall(basename(p))[0][1:-1] == well_id,
            overview_mips
        )
    )


def get_well_overview_segs(plate: PlateRecord,
                           well_id: str,
                           well_path_to_well_id: lambda p:
                           basename(p).split('_')[3]):
    seg_mips = glob(
        join(plate.experiment.root_dir, plate.plate_id,
             "TIF_OVR_MIP_SEG",
             "obj_v0.3",
             "*.tif")
    )
    return list(
        filter(lambda p: well_path_to_well_id(p) == well_id, seg_mips)
    )
