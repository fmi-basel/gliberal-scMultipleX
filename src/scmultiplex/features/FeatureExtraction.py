from typing import List

import pandas as pd
from faim_hcs.records.WellRecord import WellRecord
from skimage.measure import regionprops


def extract_2d_ovr(
    well: WellRecord, ovr_channel: str, ovr_seg_img, touching_labels: List[str]
):
    df_ovr = pd.DataFrame()
    ovr_features = regionprops(ovr_seg_img)

    for obj in ovr_features:
        organoid_id = "object_" + str(obj["label"])
        row = {
            "hcs_experiment": well.plate.experiment.name,
            "plate_id": well.plate.plate_id,
            "well_id": well.well_id,
            "organoid_id": organoid_id,
            "segmentation_ovr": well.segmentations[ovr_channel],
            "flag_tile_border": organoid_id
            in touching_labels,  # TRUE (1) means organoid is touching a tile border
            "x_pos_pix_global": obj["centroid"][1],
            "y_pos_pix_global": obj["centroid"][0],
            "area_pix_global": obj["area"],
        }

        df_ovr = pd.concat(
            [df_ovr, pd.DataFrame.from_records([row])], ignore_index=True
        )
    return df_ovr
