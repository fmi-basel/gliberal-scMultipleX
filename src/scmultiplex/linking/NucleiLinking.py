# Copyright (C) 2023 Friedrich Miescher Institute for Biomedical Research

##############################################################################
#                                                                            #
# Author: Nicole Repina              <nicole.repina@fmi.ch>                  #
# Author: Tim-Oliver Buchholz        <tim-oliver.buchholz@fmi.ch>            #
# Author: Enrico Tagliavini          <enrico.tagliavini@fmi.ch>              #
#                                                                            #
##############################################################################

from os.path import join

import numpy as np
import pandas as pd
from platymatch.utils.utils import generate_affine_transformed_image
from skimage.io import imsave

from scmultiplex.faim_hcs.records.OrganoidRecord import OrganoidRecord
from scmultiplex.linking.NucleiLinkingFunctions import (
    relabel_RX_numpy,
    run_affine,
    run_ffd,
)


def load_organoid_measurement(organoid: OrganoidRecord, org_seg_ch):
    df_ovr = organoid.well.get_measurement(f"regionprops_ovr_{org_seg_ch}")
    df_ovr = df_ovr.set_index("org_id")
    df_org = organoid.get_measurement(f"regionprops_org_{org_seg_ch}")
    return df_ovr, df_org


def load_linking_data(organoid: OrganoidRecord, rx_name: str, org_seg_ch):
    link_org = organoid.well.get_measurement(
        f"linking_ovr_{org_seg_ch}_{rx_name}toR0"
    )
    link_org_dict = link_org.set_index("R0_label").T.to_dict("index")["RX_label"]
    return link_org, link_org_dict


def link_affine(
    RX_numpy,
    R0_numpy,
    RX_obj,
    R0_obj,
    plate_id,
    well_id,
    ransac_iterations=4000,
    icp_iterations=50,
):
    """
    Run affine transformation using PlatyMatch and format result to pandas df
    """
    (affine_matches, transform_affine) = run_affine(
        RX_numpy, R0_numpy, ransac_iterations, icp_iterations
    )
    affine_matches = pd.DataFrame(
        affine_matches,
        columns=["R0_nuc_id", "RX_nuc_id", "R0RX_pixdist_affine", "confidence"],
    )
    affine_matches["R0_organoid_id"] = R0_obj
    affine_matches["RX_organoid_id"] = RX_obj
    affine_matches["plate_id"] = plate_id
    affine_matches["well_id"] = well_id

    return affine_matches, transform_affine


def link_ffd(
    RX_numpy,
    R0_numpy,
    RX_raw,
    R0_raw,
    R0_seg,
    RX_seg,
    RX_obj,
    R0_obj,
    plate_id,
    well_id,
    transform_affine,
):
    """
    Apply affine transformation, run ffd transformation using PlatyMatch, and format result to pandas df
    """
    # generate transformed affine image
    (
        moving_transformed_affine_raw_image,
        moving_transformed_affine_label_image,
    ) = generate_affine_transformed_image(
        transform_matrix=transform_affine,
        fixed_raw_image=R0_raw,
        moving_raw_image=RX_raw,
        moving_label_image=RX_seg,
    )
    # run ffd matching
    # for now only use ffd_matches result, do not save transform or transformed image
    (ffd_matches, transform_ffd, transformed_ffd_label_image) = run_ffd(
        RX_numpy,
        R0_numpy,
        moving_transformed_affine_raw_image,
        moving_transformed_affine_label_image,
        R0_raw,
        R0_seg,
    )

    ffd_matches = pd.DataFrame(
        ffd_matches, columns=["R0_nuc_id", "RX_nuc_id", "R0RX_pixdist_ffd"]
    )
    ffd_matches["R0_organoid_id"] = R0_obj
    ffd_matches["RX_organoid_id"] = RX_obj
    ffd_matches["plate_id"] = plate_id
    ffd_matches["well_id"] = well_id

    return ffd_matches


def apply_transform(
    transform_affine, organoid_R0, organoid_RX, RX_seg, RX_savepath, nuc_seg_ch
):

    if transform_affine is None:
        return None

    for channel in organoid_RX.raw_files:
        R0_raw = organoid_R0.get_raw_data(
            nuc_seg_ch
        )  # doesn't matter which channel; can load seg channel here
        RX_raw = organoid_RX.get_raw_data(channel)

        transformed_affine_raw_image = generate_affine_transformed_image(
            transform_affine, R0_raw, RX_raw, RX_seg
        )[0]

        print(
            transformed_affine_raw_image.dtype,
            np.amin(transformed_affine_raw_image),
            np.amax(transformed_affine_raw_image),
        )

        # Save transformed RX image
        path = join(RX_savepath, channel + "_affine_transformed.tif")
        imsave(
            path, transformed_affine_raw_image.astype(np.uint16), check_contrast=False
        )

    return transformed_affine_raw_image


def link_nuclei(organoid, segname, rx_name, RX, z_anisotropy, org_seg_ch, nuc_seg_ch):
    """
    Run PlatyMatch linking using Prefect/FAIM-HCS data structure
    """
    R0_obj = organoid.organoid_id
    R0_id = int(R0_obj.rpartition("_")[2])
    well_id = organoid.well.well_id
    plate_id = organoid.well.plate.plate_id
    names = ["R0", rx_name]

    R0_df_ovr, R0_df_org = load_organoid_measurement(organoid, org_seg_ch)

    link_org, link_org_dict = load_linking_data(organoid, rx_name, org_seg_ch)

    if R0_id in link_org_dict:
        if not R0_df_ovr.loc[R0_id, "flag_tile_border"]:
            if R0_df_org["abs_min"][0] != 0:
                try:
                    R0_df = organoid.get_measurement(
                        f"regionprops_nuc_{nuc_seg_ch}"
                    )
                    R0_raw = organoid.get_raw_data(org_seg_ch)
                    R0_seg = organoid.get_segmentation(segname)
                except Exception as e:
                    print(e)
                    return None, None, None

                RX_id = link_org_dict[R0_id]
                RX_obj = "object_" + str(RX_id)

                RX_df = (
                    RX.plates[plate_id]
                    .wells[well_id]
                    .organoids[RX_obj]
                    .get_measurement(f"regionprops_nuc_{nuc_seg_ch}")
                )
                RX_raw = (
                    RX.plates[plate_id]
                    .wells[well_id]
                    .organoids[RX_obj]
                    .get_raw_data(org_seg_ch)
                )
                RX_seg = (
                    RX.plates[plate_id]
                    .wells[well_id]
                    .organoids[RX_obj]
                    .get_segmentation(segname)
                )

                RX_savepath = (
                    RX.plates[plate_id].wells[well_id].organoids[RX_obj].organoid_dir
                )

                # N x 5 (first column is ids, last column is size)
                R0_numpy = R0_df[
                    [
                        "nuc_id",
                        "x_pos_pix",
                        "y_pos_pix",
                        "z_pos_pix_scaled",
                        "volume_pix",
                    ]
                ].to_numpy()
                RX_numpy = RX_df[
                    [
                        "nuc_id",
                        "x_pos_pix",
                        "y_pos_pix",
                        "z_pos_pix_scaled",
                        "volume_pix",
                    ]
                ].to_numpy()

                # output of feature extraction z-centroid is scaled by z-anisotropy
                # output of feature extraction volume is scaled by z-anisotropy
                # however since platymatch uses label images for linking, must remove this scaling here
                # so, divide by same z-anisotropy used during feature extr so centroids and volumes match label image!
                R0_numpy[:, 3] /= z_anisotropy
                R0_numpy[:, 4] /= z_anisotropy
                RX_numpy[:, 3] /= z_anisotropy
                RX_numpy[:, 4] /= z_anisotropy

                transform_affine = None

                if (R0_numpy.shape[0] > 4) and (RX_numpy.shape[0] > 4):
                    print("matching of", R0_obj, "and", RX_obj, "in", plate_id, well_id)

                    # affine linking
                    try:
                        (affine_matches, transform_affine) = link_affine(
                            RX_numpy,
                            R0_numpy,
                            RX_obj,
                            R0_obj,
                            plate_id,
                            well_id,
                            ransac_iterations=4000,
                            icp_iterations=50,
                        )

                        # Save measurement into the organoid directory.
                        name = "linking_nuc_affine_" + names[1] + "to" + names[0]
                        path = join(organoid.organoid_dir, name + ".csv")
                        affine_matches.to_csv(path, index=False)  # saves csv

                        # Add the measurement to the faim-hcs datastructure and save.
                        organoid.add_measurement(name, path)
                        organoid.save()  # updates json file

                        # Save affine transformation matrix to organoid directory
                        path = join(RX_savepath, "affine.npy")
                        np.save(path, transform_affine, allow_pickle=False)
                        # to load from saved file, use np.load('filepath')

                        # Relabel RX segmentation images to match R0 labeling
                        RX_numpy_matched = relabel_RX_numpy(
                            RX_seg,
                            affine_matches,
                            moving_colname="RX_nuc_id",
                            fixed_colname="R0_nuc_id",
                            daskarr=False,
                        )

                        # Save relabelled RX image
                        path = join(RX_savepath, "relabelled.tif")
                        imsave(
                            path,
                            RX_numpy_matched.astype(np.int16),
                            check_contrast=False,
                        )

                    except Exception as e:
                        print(R0_obj, RX_obj, e)

                    # ffd linking
                    try:
                        ffd_matches = link_ffd(
                            RX_numpy,
                            R0_numpy,
                            RX_raw,
                            R0_raw,
                            R0_seg,
                            RX_seg,
                            RX_obj,
                            R0_obj,
                            plate_id,
                            well_id,
                            transform_affine,
                        )

                        # Save measurement into the organoid directory.
                        name = "linking_nuc_ffd_" + names[1] + "to" + names[0]
                        path = join(organoid.organoid_dir, name + ".csv")
                        ffd_matches.to_csv(path, index=False)  # saves csv

                        # Add the measurement to the faim-hcs datastructure and save.
                        organoid.add_measurement(name, path)
                        organoid.save()  # updates json file

                    except Exception as e:
                        print(R0_obj, RX_obj, e)

    return (
        transform_affine,
        organoid,
        RX.plates[plate_id].wells[well_id].organoids[RX_obj],
        RX_seg,
        RX_savepath,
    )
