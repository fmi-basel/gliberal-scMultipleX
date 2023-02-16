from os.path import join

import pandas as pd
from faim_hcs.records.OrganoidRecord import OrganoidRecord

from scmultiplex.platymatch.run_platymatch import runAffine, runFFD


def load_organoid_measurement(organoid: OrganoidRecord):
    df_ovr = organoid.well.get_measurement("regionprops_ovr_C01")
    df_ovr = df_ovr.set_index("org_id")
    df_org = organoid.get_measurement("regionprops_org_C01")
    return df_ovr, df_org


def load_linking_data(organoid: OrganoidRecord, rx_name: str):
    link_org = organoid.well.get_measurement("linking_ovr_NEW_C01_" + rx_name + "toR0")
    link_org_dict = link_org.set_index("R0_label").T.to_dict("index")["RX_label"]
    return link_org, link_org_dict


def link_nuclei(organoid, ovr_channel, segname, rx_name, RX, z_anisotropy):
    R0_obj = organoid.organoid_id
    R0_id = int(R0_obj.rpartition("_")[2])
    well_id = organoid.well.well_id
    plate_id = organoid.well.plate.plate_id
    names = ["R0", rx_name]

    R0_df_ovr, R0_df_org = load_organoid_measurement(organoid)

    link_org, link_org_dict = load_linking_data(organoid, rx_name)

    if R0_id in link_org_dict:
        if not R0_df_ovr.loc[R0_id, "flag_tile_border"]:
            if R0_df_org["abs_min"][0] != 0:
                try:
                    R0_df = organoid.get_measurement("regionprops_nuc_C01")
                    R0_raw = organoid.get_raw_data(ovr_channel)
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
                    .get_measurement("regionprops_nuc_C01")
                )
                RX_raw = (
                    RX.plates[plate_id]
                    .wells[well_id]
                    .organoids[RX_obj]
                    .get_raw_data(ovr_channel)
                )
                RX_seg = (
                    RX.plates[plate_id]
                    .wells[well_id]
                    .organoids[RX_obj]
                    .get_segmentation(segname)
                )

                # N x 5 (first column is ids, last column is size)
                R0_numpy = R0_df[
                    ["nuc_id", "x_pos_vox", "y_pos_vox", "z_pos_vox", "volume_pix"]
                ].to_numpy()
                RX_numpy = RX_df[
                    ["nuc_id", "x_pos_vox", "y_pos_vox", "z_pos_vox", "volume_pix"]
                ].to_numpy()
                # Divid by z voxel anisotropy so that coordinates match label image!
                R0_numpy[:, 3] *= z_anisotropy
                R0_numpy[:, 4] *= z_anisotropy
                RX_numpy[:, 3] *= z_anisotropy
                RX_numpy[:, 4] *= z_anisotropy

                if (R0_numpy.shape[0] > 4) and (RX_numpy.shape[0] > 4):
                    ransac_iterations = 4000
                    icp_iterations = 50

                    print("matching of", R0_obj, "and", RX_obj, "in", plate_id, well_id)

                    try:
                        # affine_matches, ffd_matches = runPM(RX_numpy, R0_numpy, ransac_iterations, icp_iterations, RX_raw, R0_raw, RX_seg, R0_seg, "savename")
                        (
                            affine_matches,
                            transform_matrix_combined,
                            confidence,
                        ) = runAffine(
                            RX_numpy, R0_numpy, ransac_iterations, icp_iterations
                        )

                        affine_matches = pd.DataFrame(
                            affine_matches,
                            columns=["R0_nuc_id", "RX_nuc_id", "confidence"],
                        )
                        affine_matches["R0_organoid_id"] = R0_obj
                        affine_matches["RX_organoid_id"] = RX_obj
                        affine_matches["plate_id"] = plate_id
                        affine_matches["well_id"] = well_id

                        # Save measurement into the organoid directory.
                        name = "linking_nuc_affine_" + names[1] + "to" + names[0]
                        path = join(organoid.organoid_dir, name + ".csv")
                        affine_matches.to_csv(path, index=False)  # saves csv

                        # Add the measurement to the faim-hcs datastructure and save.
                        organoid.add_measurement(name, path)
                        organoid.save()  # updates json file

                    except Exception as e:
                        print(R0_obj, RX_obj, e)

                    try:
                        ffd_matches = runFFD(
                            R0_numpy,
                            RX_raw,
                            R0_raw,
                            RX_seg,
                            R0_seg,
                            transform_matrix_combined,
                        )

                        ffd_matches = pd.DataFrame(
                            ffd_matches, columns=["R0_nuc_id", "RX_nuc_id"]
                        )
                        # ffd_matches = pd.DataFrame(ffd_matches, columns=['R0_nuc_id', 'RX_nuc_id', 'confidence'])
                        ffd_matches["R0_organoid_id"] = R0_obj
                        ffd_matches["RX_organoid_id"] = RX_obj
                        ffd_matches["plate_id"] = plate_id
                        ffd_matches["well_id"] = well_id

                        # Save measurement into the organoid directory.
                        name = "linking_nuc_ffd_" + names[1] + "to" + names[0]
                        path = join(organoid.organoid_dir, name + ".csv")
                        ffd_matches.to_csv(path, index=False)  # saves csv

                        # Add the measurement to the faim-hcs datastructure and save.
                        organoid.add_measurement(name, path)
                        organoid.save()  # updates json file

                    except Exception as e:
                        print(R0_obj, RX_obj, e)
