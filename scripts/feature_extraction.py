from typing import List

from faim_hcs.records.WellRecord import WellRecord
from prefect import Flow, Parameter, task, unmapped

from src.scmultiplex.features.FeatureExtraction import extract_2d_ovr
from src.scmultiplex.features.FeatureFunctions import flag_touching
from src.scmultiplex.utils.exclude_utils import exclude_conditions

# wrap a function in a prefect task
from src.scmultiplex.utils.load_utils import load_experiment, load_ovr
from src.scmultiplex.utils.save_utils import save_to_well


@task()
def load_task(exp_path: str, excluded_plates: List[str], excluded_wells: List[str]):
    exp = load_experiment(exp_path)
    wells = exclude_conditions(exp, excluded_plates, excluded_wells)
    return exp, wells


# def exclude_conditions_task(exp: Experiment, excluded_plates: List[str], excluded_wells: List[str]):
#     return exclude_conditions_task(exp, excluded_plates, excluded_wells)


@task()
def feature_extraction_ovr_task(well: WellRecord, name_ovr: str):
    ovr_seg_img, ovr_seg_tiles = load_ovr(well, ovr_channel)
    touching_labels_lst = flag_touching(ovr_seg_img, ovr_seg_tiles)
    df_ovr = extract_2d_ovr(well, ovr_channel, ovr_seg_img, touching_labels_lst)
    save_to_well(well, name_ovr, df_ovr)


# all feature extraction in one flow because writing to the same json file
with Flow("Feature-Extraction") as flow:
    # parameters
    exp_path = Parameter("exp_path")
    excluded_plates = Parameter("excluded_plates", default=[])
    excluded_wells = Parameter("excluded_wells", default=[])
    ovr_channel = Parameter("ovr_channel", default="C01")
    name_ovr = Parameter("name_ovr", default="regionprops_ovr_" + str(ovr_channel))

    # flow
    exp, wells = load_task(exp_path, excluded_plates, excluded_wells)
    # how to only iterate over wells?
    feature_extraction_ovr_task.map(wells, unmapped(ovr_channel))

    #

    # loaded = load_ovr.map(wells, unmapped(ovr_channel))

    # for well in wells:
    # get overview image -- load images?
    # binarize tile img and find touching labels
    # feature extraction with regionprops
    # add row for desired features to make df
    # save into well directory
