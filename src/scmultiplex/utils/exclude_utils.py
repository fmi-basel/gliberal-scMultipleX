from typing import List

from faim_hcs.hcs.Experiment import Experiment


def exclude_conditions(
    exp: Experiment, excluded_plates: List[str], excluded_wells: List[str]
):
    """
    exclude entire plate from experiment based on plate name
    exclude wells from entire experiment (all plates) based on well ID
    """
    exp.only_iterate_over_wells(True)
    exp.reset_iterator()

    wells = []

    for well in exp:
        if well.plate.plate_id not in excluded_plates:
            if well.well_id not in excluded_wells:
                wells.append(well)

    return wells


# def select_wells(
#         exp: Experiment,
#         excluded: List[str]
# ):
#     exp.only_iterate_over_wells(True)
#     exp.reset_iterator()
#
#     wells = []
#
#     for well in exp:
#         if well.well_id not in excluded:
#             wells.append(well.well_id)
#
#     return wells
