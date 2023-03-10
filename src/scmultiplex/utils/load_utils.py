# Copyright (C) 2023 Friedrich Miescher Institute for Biomedical Research

##############################################################################
#                                                                            #
# Author: Nicole Repina              <nicole.repina@fmi.ch>                  #
# Author: Tim-Oliver Buchholz        <tim-oliver.buchholz@fmi.ch>            #
# Author: Enrico Tagliavini          <enrico.tagliavini@fmi.ch>              #
#                                                                            #
##############################################################################

from faim_hcs.hcs.Experiment import Experiment
from faim_hcs.records.WellRecord import WellRecord


def load_experiment(exp_path):
    exp = Experiment()
    exp.load(exp_path)
    return exp


# load ovr images
def load_ovr(well: WellRecord, ovr_channel: str):
    ovr_seg = well.get_segmentation(ovr_channel)  # this is the seg image

    if ovr_seg is not None:
        dims = len(ovr_seg.shape)

        # if it is a 3D image, contains tiling from Drogon
        if dims == 3:
            ovr_seg_img = ovr_seg[0]  # first image is label map
            ovr_seg_tiles = ovr_seg[1]  # second image is drogon tiling
        elif dims == 2:
            ovr_seg_img = ovr_seg
            ovr_seg_tiles = well.get_raw_data(ovr_channel)[
                1
            ]  # load raw MIP image where second image is Drogon tiling
        else:
            raise NotImplementedError("only supporting 2D or 3D images")

        # binarize tile image
        ovr_seg_tiles[ovr_seg_tiles > 0] = 1

        return ovr_seg_img, ovr_seg_tiles
    else:
        return None, None
