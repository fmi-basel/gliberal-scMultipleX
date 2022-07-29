from typing import List

import prefect
from faim_hcs.records.WellRecord import WellRecord
from prefect import Flow, Parameter, task, unmapped
from prefect.executors import LocalDaskExecutor
from prefect.run_configs import LocalRun
from prefect.storage import Local

from scmultiplex.features.FeatureExtraction import extract_2d_ovr
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
def feature_extraction_ovr_task(well: WellRecord, ovr_channel: str, name_ovr: str):
    logger = prefect.context.get("logger")
    ovr_seg_img, ovr_seg_tiles = load_ovr(well, ovr_channel)

    if ovr_seg_img is None and ovr_seg_tiles is None:
        logger.warning(f"ovr_seg does not exists. Skipping {well.well_id}.")
    else:
        touching_labels_lst = flag_touching(ovr_seg_img, ovr_seg_tiles)
        df_ovr = extract_2d_ovr(well, ovr_channel, ovr_seg_img, touching_labels_lst)
        save_to_well(well, name_ovr + ovr_channel, df_ovr)


# all feature extraction in one flow because writing to the same json file
with Flow(
    "Feature-Extraction",
    storage=Local(
        directory="/home/tibuch/Prefect_Flows/scMultipleX_flows",
        stored_as_script=True,
        path="/home/tibuch/Gitrepos/gliberal-scMultipleX/scripts"
        "/prefect/feature_extraction.py",
    ),
    executor=LocalDaskExecutor(),
    run_config=LocalRun(),
) as flow:
    # parameters
    exp_path = Parameter("exp_path", default="/path/to/exp")
    excluded_plates = Parameter("excluded_plates", default=[])
    excluded_wells = Parameter("excluded_wells", default=[])
    ovr_channel = Parameter("ovr_channel", default="C01")
    name_ovr = Parameter("name_ovr", default="regionprops_ovr_")

    # flow
    exp, wells = load_task(exp_path, excluded_plates, excluded_wells)
    # how to only iterate over wells?
    feature_extraction_ovr_task.map(wells, unmapped(ovr_channel), unmapped(name_ovr))

    #

    # loaded = load_ovr.map(wells, unmapped(ovr_channel))

    # for well in wells:
    # get overview image -- load images?
    # binarize tile img and find touching labels
    # feature extraction with regionprops
    # add row for desired features to make df
    # save into well directory
