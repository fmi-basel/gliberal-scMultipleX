# Copyright 2024 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Nicole Repina <nicole.repina@fmi.ch>
#

"""
Calculate spherical harmonics of 3D input label image using aics_shparam.
"""

import anndata as ad
import dask.array as da
import logging
import numpy as np
import os
import pandas as pd
import zarr

from fractal_tasks_core.tables import write_table
from fractal_tasks_core.tasks.io_models import InitArgsRegistrationConsensus
from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.roi import (
    check_valid_ROI_indices,
    convert_indices_to_regions,
    convert_ROI_table_to_indices,
    load_region)
from pydantic.decorator import validate_arguments
from typing import Any

from scmultiplex.fractal.fractal_helper_functions import format_roi_table
from scmultiplex.meshing.MeshFunctions import export_stl_polydata
from scmultiplex.aics_shparam import shparam

logger = logging.getLogger(__name__)


@validate_arguments
def spherical_harmonics_from_labelimage(
        *,
        # Fractal arguments
        zarr_url: str,
        init_args: InitArgsRegistrationConsensus,
        # Task-specific arguments
        label_name: str = "org_3d",
        roi_table: str = "org_ROI_table_3d",
        level: int = 0,
        lmax: int = 2,
        save_mesh: bool = True,

) -> dict[str, Any]:
    """
    Calculate spherical harmonics of 3D input label image using aics_shparam.

    This task consists of 5 parts:

    1. Load 3D label image based on provided label name and ROI table
    2. Compute 3D mesh with vtkContourFilter using aics_shparam functions
    3. Compute spherical harmonics of mesh using aics_shparam
    4. Compute reconstruction error (mse) of computed harmonics
    5. Optionally generate reconstructed mesh from the calculated harmonics
    6. Output: save the (1) spherical harmonic coefficients and mse error as feature table
        (2) reconstructed meshes (.stl) per object id in a new meshes folder within zarr structure

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            Refers to the zarr_url of the reference acquisition.
            (standard argument for Fractal tasks, managed by Fractal server).
        init_args: Intialization arguments provided by
            `init_group_by_well_for_multiplexing`. It contains the
            zarr_url_list listing all the zarr_urls in the same well as the
            zarr_url of the reference acquisition that are being processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        label_name: Label for which spherical harmonics are calculated
        roi_table: Name of the ROI table over which the task loops to
            calculate the registration. e.g. consensus object table 'org_ROI_table_consensus'
        level: Pyramid level of the labels to register. Choose `0` to
            process at full resolution.
        lmax: Maximum degree of the spherical harmonics coefficients
        save_mesh: if True, saves the vtk mesh on disk in subfolder 'meshes'. Filename corresponds to object label id

    """
    logger.info(
        f"Running for {zarr_url=}. \n"
        f"Calculating spherical harmonics for objects in {label_name=} with ROI table "
        f"{roi_table=}."
    )

    # Lazily load zarr array for reference cycle
    # load well image as dask array e.g. for nuclear segmentation
    r0_dask = da.from_zarr(f"{zarr_url}/labels/{label_name}/{level}")

    # Read ROIs of objects
    r0_adata = ad.read_zarr(f"{zarr_url}/tables/{roi_table}")

    # Read Zarr metadata
    r0_ngffmeta = load_NgffImageMeta(f"{zarr_url}/labels/{label_name}")
    r0_xycoars = r0_ngffmeta.coarsening_xy # need to know when building new pyramids
    r0_pixmeta = r0_ngffmeta.get_pixel_sizes_zyx(level=level)

    # Create list of indices for 3D ROIs spanning the entire Z direction
    r0_idlist = convert_ROI_table_to_indices(
        r0_adata,
        level=level,
        coarsening_xy=r0_xycoars,
        full_res_pxl_sizes_zyx=r0_pixmeta,
    )

    check_valid_ROI_indices(r0_idlist, roi_table)

    r0_labels = r0_adata.obs_vector('label')
    # initialize variables
    compute = True  # convert to numpy array from dask

    if len(r0_idlist) == 0:
        logger.warning("Well contains no objects")

    df_coeffs = []
    # for every object in ROI table...
    for row in r0_adata.obs_names:
        row_int = int(row)
        r0_org_label = r0_labels[row_int]
        region = convert_indices_to_regions(r0_idlist[row_int])

        # load label image for object
        seg = load_region(
            data_zyx=r0_dask,
            region=region,
            compute=compute,
        )

        #Mask to remove any neighboring organoids
        #TODO: Improve this during the NGIO refactor, very basic masking here to select the desired organoid label
        seg[seg != float(r0_org_label)] = 0

        ##############
        # Compute mesh and spherical harmonics coefficients  ###
        ##############

        spacing = tuple(np.array(r0_pixmeta) / r0_pixmeta[1])  # z,y,x e.g. (2.78, 1, 1)
        print('spacing', spacing)

        ((coeffs, grid_rec),
         (image_, mesh_polydata_organoid, grid, transform)) = shparam.get_shcoeffs(image=seg,
                                                                                   lmax=lmax,
                                                                                   sigma=5,
                                                                                   spacing=spacing)
        coeffs.update({'label': r0_org_label})
        df_coeffs.append(coeffs)

        ##############
        # Save mesh (optional) ###
        ##############

        if save_mesh:
            save_mesh_path = f"{zarr_url}/meshes/{roi_table}_shaics"
            os.makedirs(save_mesh_path, exist_ok=True)
            # save name is the organoid label id
            save_name = f"{int(r0_org_label)}.stl"
            export_stl_polydata(os.path.join(save_mesh_path, save_name), mesh_polydata_organoid)


    ##############
    # Save spherical harmonics as measurement table  ###
    ##############
    df_coeffs = pd.DataFrame(df_coeffs)

    # follows similar logic as feature extraction task
    if not df_coeffs.empty:
        measurement_table = format_roi_table([df_coeffs])
    else:
        # Create empty anndata table
        measurement_table = ad.AnnData()

    # Write to zarr group
    image_group = zarr.group(f"{zarr_url}")
    table_attrs = {
        "type": "feature_table",
        "fractal_table_version" : "1",
        "region": {"path": f"../labels/{label_name}"},
        "instance_key": "label",
    }
    output_table_name = label_name + '_harmonics'
    write_table(
        image_group,
        output_table_name,
        measurement_table,
        overwrite=True,
        table_attrs=table_attrs,
    )

    logger.info(f"End spherical_harmonics_from_labelimage task for {zarr_url=}")

    return {}


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=spherical_harmonics_from_labelimage,
        logger_name=logger.name,
    )
