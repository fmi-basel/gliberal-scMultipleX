# Copyright 2023 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Nicole Repina <nicole.repina@fmi.ch>
#

"""
Calculates 3D surface mesh of parent object (e.g. tissue, organoid)
from 3D cell-level segmentation of children (e.g. nuclei)
"""
from pathlib import Path
from typing import Any
from typing import Sequence

import anndata as ad
import dask.array as da
import logging
import numpy as np
import os
import pandas as pd

import zarr
from fractal_tasks_core.labels import prepare_label_group
from fractal_tasks_core.pyramids import build_pyramid
from fractal_tasks_core.upscale_array import upscale_array
from pydantic.decorator import validate_arguments

from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.roi import (
    check_valid_ROI_indices,
    convert_indices_to_regions,
    convert_ROI_table_to_indices,
    load_region)
from scipy.ndimage import binary_fill_holes, binary_erosion

from skimage.feature import canny
from skimage.filters import gaussian
from skimage.measure import label, regionprops_table
from skimage.morphology import disk, remove_small_objects
from skimage.segmentation import expand_labels

from scmultiplex.fractal.FractalHelperFunctions import get_zattrs

from scmultiplex.meshing.FilterFunctions import filter_small_sizes_per_round, equivalent_diam
from scmultiplex.meshing.MeshFunctions import labels_to_mesh, export_vtk_polydata

logger = logging.getLogger(__name__)


@validate_arguments
def surface_mesh(
        *,
        # Fractal arguments
        input_paths: Sequence[str],
        output_path: str,
        component: str,
        metadata: dict[str, Any],
        # Task-specific arguments
        label_name: str = "nuc",
        label_name_obj: str = "org_consensus",
        roi_table: str = "org_ROI_table_consensus",
        reference_cycle: int = 0,
        level: int = 0,
        expandby_factor: float = 0.6,
        sigma_factor: float = 5,
        canny_threshold: float = 0.3,
        mask_by_parent: bool = True,
        save_mesh: bool = True,
        save_labels: bool = True,

) -> dict[str, Any]:
    """
    Calculate 3D surface mesh of parent object (e.g. tissue, organoid)
    from 3D cell-level segmentation of children (e.g. nuclei)

    This task consists of 4 parts:

    1. Load the sub-object (e.g. nuc) segmentation images for each object of a given reference round; skip other rounds.
        Select sub-objects (e.g. nuc) that belong to parent object region by masking by parent.
        Filter the sub-objects to remove small debris that was segmented.
    2. Perform label fusion and edge detection to generate surface label image.
    3. Calculate surface mesh of label image using marching cubes algorithm.
    4. Output: save the (1) meshes (.vtp) per object id in meshes folder and (2) well label map as a new label in zarr.
        Note that label map output may be clipped for objects that are dense and have overlapping pixels.
        In this case, the 'winning' object in the overlap region is the one with higher label id.

    Parallelization level: image

    Args:
        input_paths: List of input paths where the image data is stored as
            OME-Zarrs. Should point to the parent folder containing one or many
            OME-Zarr files, not the actual OME-Zarr file. Example:
            `["/some/path/"]`. This task only supports a single input path.
            (standard argument for Fractal tasks, managed by Fractal server).
        output_path: This parameter is not used by this task.
            (standard argument for Fractal tasks, managed by Fractal server).
        component: Path to the OME-Zarr image in the OME-Zarr plate that is
            processed. Example: `"some_plate.zarr/B/03/1"`.
            (standard argument for Fractal tasks, managed by Fractal server).
        metadata: This parameter is not used by this task.
            (standard argument for Fractal tasks, managed by Fractal server).
        label_name: Label name that will be used for surface estimation, e.g. `nuc`.
        label_name_obj: Label name of segmented objects that is parent of
            label_name e.g. `org_consensus`.
        roi_table: Name of the ROI table over which the task loops to
            calculate the registration. e.g. consensus object table 'org_ROI_table_consensus'
        reference_cycle: Which cycle to use for surface mesh calculation. Defaults to 0,
            which is the first OME-Zarr image in the well (usually the first
            cycle that was provided).
        level: Pyramid level of the labels to register. Choose `0` to
            process at full resolution.
        expandby_factor: multiplier that specifies pixels by which to expand each nuclear mask for merging,
            float in range [0,1 or higher], e.g. 0.2 means that 20% of mean of nuclear equivalent diameter is used.
        sigma_factor: float that specifies sigma (standard deviation) for Gaussian kernel. Higher
            values correspond to more blurring. Recommended range 1-8.
        canny_threshold: image values below this threshold are set to 0 after Gaussian blur. float in range [0,1].
            Higher values result in tighter fit of mesh to nuclear surface
        mask_by_parent: if True, nuclei are masked by parent object (e.g. organoid) to only select nuclei
            belonging to parent. Recommended to set to True when iterating over object (e.g. organoid) ROIs.
        save_mesh: if True, saves the vtk mesh on disk in subfolder 'meshes'. Filename corresponds to object label id
        save_labels: if True, saves the calculated 3D label map as label map in 'labels' with suffix '_3d'

    """
    logger.info(
        f"Running for {input_paths=}, {component=}. \n"
        f"Calculating surface mesh per {roi_table=} for "
        f"{label_name=}."
    )
    # Set OME-Zarr paths
    input_zarr_path = Path(input_paths[0]) / component
    # r0_zarr_path = input_zarr_path.parent / str(reference_cycle)

    # If the task is run for the any cycle that is not reference, exit
    # TODO: Improve the input for this: Can we filter components to not
    current_cycle = input_zarr_path.name
    if current_cycle != str(reference_cycle):
        logger.info(
            f"Skipping cycle {current_cycle} "
        )
        return {}

    else:
        logger.info(
            "Surface mesh calculation is running for "
            f"cycle {current_cycle}"
        )

    # Lazily load zarr array for reference cycle
    # load well image as dask array e.g. for nuclear segmentation
    r0_dask = da.from_zarr(f"{input_zarr_path}/labels/{label_name}/{level}")

    # Read ROIs
    r0_adata = ad.read_zarr(f"{input_zarr_path}/tables/{roi_table}")

    # Read Zarr metadata
    r0_ngffmeta = load_NgffImageMeta(f"{input_zarr_path}/labels/{label_name}")
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

    if len(r0_idlist) == 0:
        logger.warning("Well contains no objects")

    # initialize new zarr for 3d object label image
    # save as same dimensions as nuclear labels from which they are calculated
    if save_labels:
        shape = r0_dask.shape
        chunks = r0_dask.chunksize
        label_dtype = np.uint32
        output_label_name = label_name_obj + '_3d'
        store = zarr.storage.FSStore(f"{input_zarr_path}/labels/{output_label_name}/0")

        if len(shape) != 3 or len(chunks) != 3 or shape[0] == 1:
            raise ValueError('Expecting 3D image')

        # Add metadata to labels group
        # Get the label_attrs correctly
        # Note that the new label metadata matches the nuc metadata
        label_attrs = get_zattrs(zarr_url=f"{input_zarr_path}/labels/{label_name}")
        _ = prepare_label_group(
            image_group=zarr.group(input_zarr_path),
            label_name=output_label_name,
            overwrite=True,
            label_attrs=label_attrs,
            logger=logger,
        )

        new_label3d_array = zarr.create(
            shape=shape,
            chunks=chunks,
            dtype=label_dtype,
            store=store,
            overwrite=True,
            dimension_separator="/",
        )

        logger.info(
            f"mask will have shape {shape} "
            f"and chunks {chunks}"
        )

    ##############
    # Filter nuclei by parent mask, remove small nuclei, and disconnected component analysis ###
    ##############

    # TODO consider making this section a stand-alone Fractal task that is run before multiplexed linking...
    #  ...Then clean up label masks only once, instead of repeating every time
    if mask_by_parent:
        # load well image as dask array for parent objects
        r0_dask_parent = da.from_zarr(f"{input_zarr_path}/labels/{label_name_obj}/{level}")

        # Read Zarr metadata
        r0_ngffmeta_parent = load_NgffImageMeta(f"{input_zarr_path}/labels/{label_name_obj}")
        r0_xycoars_parent = r0_ngffmeta_parent.coarsening_xy
        r0_pixmeta_parent = r0_ngffmeta_parent.get_pixel_sizes_zyx(level=level)

        r0_idlist_parent = convert_ROI_table_to_indices(
            r0_adata,
            level=level,
            coarsening_xy=r0_xycoars_parent,
            full_res_pxl_sizes_zyx=r0_pixmeta_parent,
        )

        check_valid_ROI_indices(r0_idlist_parent, roi_table)

    r0_labels = r0_adata.obs_vector('label')
    # initialize variables
    compute = True  # convert to numpy array from dask

    # for each parent object (e.g. organoid) in r0...
    for row in r0_adata.obs_names:
        row_int = int(row)
        r0_org_label = r0_labels[row_int]
        region = convert_indices_to_regions(r0_idlist[row_int])

        # load nuclear label image for object in reference round
        seg = load_region(
            data_zyx=r0_dask,
            region=region,
            compute=compute,
        )

        if mask_by_parent:
            # load object label image for object in reference round
            r0_parent = load_region(
                data_zyx=r0_dask_parent,
                region=convert_indices_to_regions(r0_idlist_parent[row_int]),
                compute=compute,
            )
            # if object segmentation was run at a different level than nuclear segmentation,
            # need to upscale arrays to match shape
            if r0_parent.shape != seg.shape:
                r0_parent = upscale_array(array=r0_parent, target_shape=seg.shape, pad_with_zeros=False)

            # mask nuclei by parent object
            r0_parent_mask = np.zeros_like(r0_parent)
            r0_parent_mask[r0_parent == int(r0_org_label)] = 1  # select only current object and binarize object mask
            seg = seg * r0_parent_mask

            # run regionprops to extract centroids and volume of each nuc label
            # note that registration is performed on unscaled image
            # TODO: consider upscaling label image prior to alignment, in cases where z-anisotropy is
            #  extreme upscaling could lead to improved performance

        r0_props = regionprops_table(label_image=seg, properties=('label', 'centroid', 'area'))  # zyx

        # output column order must be: ["label", "x_centroid", "y_centroid", "z_centroid", "volume"]
        # units are in pixels
        r0_props = (pd.DataFrame(r0_props,
                                 columns=['label', 'centroid-2', 'centroid-1', 'centroid-0', 'area'])).to_numpy()

        # discard segmentations that have a volume less than 5% of median nuclear volume (segmented debris)
        (r0_props, removed, removed_size_mean, size_mean) = (
            filter_small_sizes_per_round(r0_props, column=-1, threshold=0.05))

        logger.info(f"\nFiltered out {len(removed)} cells from object {r0_org_label} that have a "
                    f" mean volume of {removed_size_mean} and correspond to labels \n {removed}"
                   )

        # TODO add disconnected component detection here to remove nuclei that don't belong to main organoid

        ##############
        # Perform label fusion and edge detection  ###
        ##############

        # expand labels in each z-slice and combine together
        seg_fill = np.zeros_like(seg)

        # the number of pixels by which to expand is a function of average nuclear size within the organoid.
        expandby_pix = int(round(expandby_factor * equivalent_diam(size_mean)))
        if expandby_pix == 0:
            logger.warning("Equivalent diameter is 0 or negative, thus labels not expanded. Check segmentation quality")

        iterations = int(round(expandby_pix / 2))

        # loop over each zslice and expand the labels, then fill holes
        for i, zslice in enumerate(seg):
            zslice = expand_labels(zslice, expandby_pix)  # expand labels in each zslice in xy
            zslice = binary_fill_holes(zslice)  # fill holes
            # to revert the expansion, erode by half of the expanded pixels ...
            # ...(since disk(1) has a radius of 1, i.e. diameter of 2)
            seg_fill[i, :, :] = binary_erosion(zslice, disk(1), iterations=iterations)  # erode down to original size

        # 3D gaussian blur
        seg_fill_8bit = (seg_fill * 255).astype(np.uint8)

        # calculate sigma based on z,y,x pixel spacing metadata, so that sigma scales with anisotropy
        pixel_anisotropy = r0_pixmeta[0]/np.array(r0_pixmeta)  # (z, y, x) where z is normalized to 1, e.g. (1, 3, 3)
        sigma = tuple([sigma_factor * x for x in pixel_anisotropy])
        blurred = gaussian(seg_fill_8bit, sigma=sigma, preserve_range=False)

        # Canny filter to detect gaussian edges
        edges_canny = np.zeros_like(blurred)
        # threshold
        blurred[blurred < canny_threshold] = 0  # was 0.15, 0.3
        for i, zslice in enumerate(blurred):

            if np.count_nonzero(zslice) == 0:  # if all values are 0, skip this zslice
                continue
            else:
                # TODO add zero-padding here prior to edge detection? Test in jpnb first. Then remove pad.
                edges = canny(zslice)
                edges_canny[i, :, :] = binary_fill_holes(edges)

        edges_canny = (edges_canny * 255).astype(np.uint8)

        edges_canny = label(remove_small_objects(edges_canny, int(expandby_pix/2)))

        logger.info(
            f"Calculated surface mesh for object label {r0_org_label} using parameters:"
            f"\n\texpanded by {expandby_pix} pix, \n\teroded by {iterations*2} pix, "
            f"\n\tgaussian blurred with sigma = {sigma}"
        )

        ##############
        # Calculate and save mesh  ###
        ##############

        if save_mesh:
            # Make mesh with marching cubes
            spacing = tuple(np.array(r0_pixmeta) / r0_pixmeta[1])  # z,y,x e.g. (2.78, 1, 1)

            # target reduction is expressed as the fraction of the original number of triangles
            # note the actual reduction may be less depending on triangulation and topological constraints
            # target reduction is thus proportional to organoid dimensions
            mesh_polydata_organoid = labels_to_mesh(edges_canny, spacing,
                                                    smoothing_iterations=100,
                                                    pass_band_param=0.01,
                                                    target_reduction=0.98,
                                                    show_progress=False)

            save_transform_path = f"{input_zarr_path}/meshes/{label_name_obj}_from_{label_name}"
            os.makedirs(save_transform_path, exist_ok=True)
            # save name is the organoid label id
            save_name = f"{int(r0_org_label)}.vtp"
            export_vtk_polydata(os.path.join(save_transform_path, save_name), mesh_polydata_organoid)

        ##############
        # Save labels  ###
        ##############

        if save_labels:
            # store labels as new label map in zarr
            # note that pixels of overlap in the case where two meshes are touching are overwritten by the last written object
            # thus meshes are the most accurate representation of surface, labels may be cropped

            # # TODO delete
            # fake_numpy = np.zeros_like(seg)
            # fake_numpy[1,2,2] = 2
            #
            # # Compute and store 0-th level to disk
            # da.array(fake_numpy).to_zarr(
            #     url=new_label3d_array,
            #     region=region,
            #     compute=True,
            # )

            # load dask from disk, will contain rois of the previously processed objects within for loop
            new_label3d_dask = da.from_zarr(f"{input_zarr_path}/labels/{output_label_name}/0")
            # load region of current object from disk, will include any previously processed neighboring objects
            seg_ondisk = load_region(
                data_zyx=new_label3d_dask,
                region=region,
                compute=compute,
            )

            # check that dimensions of rois match
            if seg_ondisk.shape != edges_canny.shape:
                raise ValueError('Computed label image must match image dimensions of bounding box during saving')

            # check that new label map is binary
            maxvalue = np.amax(edges_canny)
            if maxvalue != 1:
                if maxvalue == 0:
                    logger.warning('Result of canny edge detection is empty')
                else: # for max values greater than 1 or less than 0
                    raise ValueError('Result of canny edge detection must be binary, check normalization')

            # convert edge detection label image value to match object label id
            edges_canny_label = edges_canny * int(r0_org_label)
            # use fmax so that if one of the elements being compared is a NaN, then the non-nan element is returned
            edges_canny_label_tosave = np.fmax(edges_canny_label, seg_ondisk)

            # Compute and store 0-th level of new 3d label map to disk
            da.array(edges_canny_label_tosave).to_zarr(
                url=new_label3d_array,
                region=region,
                compute=True,
            )

    # Starting from on-disk highest-resolution data, build and write to disk a
    # pyramid of coarser levels
    if save_labels:

        build_pyramid(
            zarrurl=f"{input_zarr_path}/labels/{output_label_name}",
            overwrite=True,
            num_levels=r0_ngffmeta.num_levels,
            coarsening_xy=r0_ngffmeta.coarsening_xy,
            chunksize=chunks,
            aggregation_function=np.max,
        )

        logger.info(f"Built a pyramid for the {input_zarr_path}/labels/{output_label_name} label image")

    logger.info(f"End surface_mesh task for {input_zarr_path}/labels/{output_label_name}")

    return {}


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task
    # from multiprocessing import freeze_support
    #
    # freeze_support()

    run_fractal_task(
        task_function=surface_mesh,
        logger_name=logger.name,
    )
