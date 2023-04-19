# Copyright (C) 2023 Friedrich Miescher Institute for Biomedical Research

##############################################################################
#                                                                            #
# Author: Nicole Repina              <nicole.repina@fmi.ch>                  #
# Author: Tim-Oliver Buchholz        <tim-oliver.buchholz@fmi.ch>            #
# Author: Enrico Tagliavini          <enrico.tagliavini@fmi.ch>              #
#                                                                            #
##############################################################################

from os.path import join

import pandas as pd
from scmultiplex.faim_hcs.records.WellRecord import WellRecord, OrganoidRecord


def save_to_record(record, name: str, df: pd.DataFrame):
    if isinstance(record, OrganoidRecord):
        path = join(record.organoid_dir, name + ".csv")
    elif isinstance(record, WellRecord):
        path = join(record.well_dir, name + ".csv")
    else:
        raise ValueError("input object type must be FAIM-HCS OrganoidRecord or WellRecord")

    # Save measurement into the record directory.
    df.to_csv(path, index=False)  # saves csv

    # Add the measurement to the faim-hcs datastructure and save.
    record.add_measurement(name, path)
    record.save()  # updates json file

    return


def save_to_well(well: WellRecord, name: str, df: pd.DataFrame):

    # Save measurement into the well directory.
    path = join(well.well_dir, name + ".csv")
    df.to_csv(path, index=False)  # saves csv

    # Add the measurement to the faim-hcs datastructure and save.
    well.add_measurement(name, path)
    well.save()  # updates json file


