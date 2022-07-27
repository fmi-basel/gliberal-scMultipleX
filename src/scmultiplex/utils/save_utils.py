from os.path import join

import pandas as pd
from faim_hcs.records.WellRecord import WellRecord


def save_to_well(well: WellRecord, name: str, df: pd.DataFrame):

    # Save measurement into the well directory.
    path = join(well.well_dir, name + ".csv")
    df.to_csv(path, index=False)  # saves csv

    # Add the measurement to the faim-hcs datastructure and save.
    well.add_measurement(name, path)
    well.save()  # updates json file
