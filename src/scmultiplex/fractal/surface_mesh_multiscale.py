# Copyright 2024 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Nicole Repina <nicole.repina@fmi.ch>
#

"""
Calculates 3D surface mesh of parent object (e.g. tissue, organoid)
from 3D cell-level segmentation of children (e.g. nuclei)
"""
import logging
import os
from typing import Any

import anndata as ad
import dask.array as da
import numpy as np
import zarr
from fractal_tasks_core.labels import prepare_label_group
from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.pyramids import build_pyramid
from fractal_tasks_core.roi import (
    array_to_bounding_box_table,
    check_valid_ROI_indices,
    convert_indices_to_regions,
    convert_ROI_table_to_indices,
    get_overlapping_pairs_3D,
    load_region,
)
from fractal_tasks_core.tables import write_table
from fractal_tasks_core.tasks.io_models import InitArgsRegistrationConsensus
from pydantic import validate_call
from scipy.ndimage import binary_erosion, binary_fill_holes
from skimage.feature import canny
from skimage.filters import gaussian
from skimage.measure import label
from skimage.morphology import disk, remove_small_objects
from skimage.segmentation import expand_labels

from scmultiplex.features.FeatureFunctions import mesh_sphericity
from scmultiplex.fractal.fractal_helper_functions import (
    convert_indices_to_origin_zyx,
    format_roi_table,
    get_zattrs,
)
from scmultiplex.meshing.FilterFunctions import (
    calculate_mean_volume,
    equivalent_diam,
    load_border_values,
    mask_by_parent_object,
    remove_border,
)
from scmultiplex.meshing.MeshFunctions import (
    export_stl_polydata,
    get_mass_properties,
    labels_to_mesh,
)

logger = logging.getLogger(__name__)


@validate_call
def surface_mesh_multiscale(
    *,
    # Fractal arguments
    zarr_url: str,
    init_args: InitArgsRegistrationConsensus,
    # Task-specific arguments
    label_name: str = "nuc",
    label_name_obj: str = "org_linked",
    roi_table: str = "org_ROI_table_linked",
    expandby_factor: float = 0.6,
    sigma_factor: float = 5,
    canny_threshold: float = 0.3,
    calculate_mesh: bool = True,
    polynomial_degree: int = 30,
    passband: float = 0.01,
    feature_angle: int = 160,
    target_reduction: float = 0.98,
    smoothing_iterations: int = 1,
    save_labels: bool = True,
) -> dict[str, Any]:
    """
    Calculate 3D surface mesh of parent object (e.g. tissue, organoid)
    from 3D cell-level segmentation of children (e.g. nuclei)

    Recommended to run on child objects that have been filtered to remove debris and disconnected components
    (e.g. following cleanup_3d_segmentation task)

    This task consists of 4 parts:

    1. Load the sub-object (e.g. nuc) segmentation images for each object of a given reference round; skip other rounds.
        Select sub-objects (e.g. nuc) that belong to parent object region by masking by parent.
    2. Perform label fusion and edge detection to generate surface label image.
    3. Calculate surface mesh of label image using vtkDiscreteFlyingEdges3D algorithm. Smoothing is applied with
        vtkWindowedSincPolyDataFilter and tuned with 4 task parameters: passband, smoothing_iterations, feature_angle,
        polynomial_degree (minimal effect).
        The number of triangles in the mesh is optionally reduced with vtkQuadricDecimation filter to form a good
        approximation to the original geometry and is tuned with task parameter target_reduction.
    4. Output: save the (1) meshes (.stl) per object id in meshes folder and (2) well label map as a new label in zarr.
        Note that label map output may be clipped for objects that are dense and have overlapping pixels.
        In this case, the 'winning' object in the overlap region is the one with higher label id.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            Refers to the zarr_url of the reference acquisition.
            (standard argument for Fractal tasks, managed by Fractal server).
        init_args: Intialization arguments provided by
            `init_group_by_well_for_multiplexing`. It contains the
            zarr_url_list listing all the zarr_urls in the same well as the
            zarr_url of the reference acquisition that are being processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        label_name: Label name of child objects that will be used for multiscale surface estimation, e.g. `nuc`.
        label_name_obj: Label name of segmented objects that are parents of
            label_name e.g. `org`.
        roi_table: Name of the ROI table that corresponds to label_name_obj. The task loops over ROIs in this table
            to load the corresponding child objects.
        expandby_factor: multiplier that specifies pixels by which to expand each nuclear mask for merging,
            float in range [0, 1 or higher], e.g. 0.2 means that 20% of mean of nuclear equivalent diameter is used.
        sigma_factor: float that specifies sigma (standard deviation, in pixels) for Gaussian kernel used for blurring
            to smoothen label image prior to edge detection. Higher values correspond to more blurring.
            Recommended range 1-8.
        canny_threshold: image values below this threshold are set to 0 after Gaussian blur. float in range [0,1].
            Higher values result in tighter fit of mesh to nuclear surface
        calculate_mesh: if True, saves the mesh as .stl on disk in meshes/[labelname] folder within zarr structure.
            Filename corresponds to object label id
        polynomial_degree: the number of polynomial degrees during surface mesh smoothing with
            vtkWindowedSincPolyDataFilter determines the maximum number of smoothing passes.
            This number corresponds to the degree of the polynomial that is used to approximate the windowed sinc
            function. Usually 10-20 iteration are sufficient. Higher values have little effect on smoothing.
            For further details see VTK vtkWindowedSincPolyDataFilter documentation.
        passband: float in range [0,2] that specifies the PassBand for the windowed sinc filter in
            vtkWindowedSincPolyDataFilter during mesh smoothing. Lower passband values produce more smoothing, due to
            filtering of higher frequencies. For further details see VTK vtkWindowedSincPolyDataFilter documentation.
        feature_angle: int in range [0,180] that specifies the feature angle for sharp edge identification used
            for vtk FeatureEdgeSmoothing. Higher values result in more smoothened edges in mesh.
            For further details see VTK vtkWindowedSincPolyDataFilter documentation.
        target_reduction: float in range [0,1]. Target reduction is used during mesh decimation via
            vtkQuadricDecimation to reduce the number of triangles in a triangle mesh, forming a good approximation to
            the original geometry. Values closer to 1 indicate larger reduction and smaller mesh file size. Note that
            target_reduction is expressed as the fraction of the original number of triangles in mesh and so is
            proportional to original mesh size. Note the actual reduction may be less depending on triangulation and
            topological constraints. For further details see VTK vtkQuadricDecimation documentation.
        smoothing_iterations: the number of iterations that mesh smoothing and decimation is run.
            If smoothing_iterations > 1, the decimated result is used as input for subsequent smoothing rounds.
            Recommended to start with 1 iteration and increase if resulting smoothing is insufficient. For each
            iteration after the first, the passband is reduced and feature_angle is increased incrementally to
            enhance smoothing. Specifically, the passband is reduced by factor 1/(2^n),
            and the feature_angle is increased by an increment of (5 * n), where n = iteration round (n=0
            is the first iteration). The target_reduction is fixed to 0.1 smoothing_iterations > 1.
            For example if user inputs (0.01, 160, 0.98) for (passband, feature_angle, target_reduction), the second
            iteration will use parameters (0.005, 165, 0.1), the third (0.0025, 170, 0.1), etc. Maximum feature_angle
            is capped at 180. Note that additional iterations usually do not significantly add to processing time as
            the number of mesh verteces is typically significantly reduced after the first decimation iteration.
        save_labels: if True, saves the calculated 3D label map as label map in 'labels' with suffix '_3d'

    """
    logger.info(
        f"Running for {zarr_url=}. \n"
        f"Calculating surface mesh per {roi_table=} for "
        f"{label_name=}."
    )

    # always use highest resolution label
    level = 0

    # Lazily load zarr array for reference cycle
    # load well image as dask array e.g. for nuclear segmentation
    r0_dask = da.from_zarr(f"{zarr_url}/labels/{label_name}/{level}")

    # Read ROIs of objects
    r0_adata = ad.read_zarr(f"{zarr_url}/tables/{roi_table}")

    # Read Zarr metadata
    r0_ngffmeta = load_NgffImageMeta(f"{zarr_url}/labels/{label_name}")
    r0_xycoars = r0_ngffmeta.coarsening_xy  # need to know when building new pyramids
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
    output_label_name = label_name_obj + "_3d"
    output_roi_table_name = roi_table + "_3d"

    if save_labels:
        shape = r0_dask.shape
        chunks = r0_dask.chunksize
        label_dtype = np.uint32
        store = zarr.storage.FSStore(f"{zarr_url}/labels/{output_label_name}/0")

        if len(shape) != 3 or len(chunks) != 3 or shape[0] == 1:
            raise ValueError("Expecting 3D image")

        # Add metadata to labels group
        # Get the label_attrs correctly
        # Note that the new label metadata matches the nuc metadata
        label_attrs = get_zattrs(zarr_url=f"{zarr_url}/labels/{label_name}")
        _ = prepare_label_group(
            image_group=zarr.group(zarr_url),
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

        logger.info(f"Mask will have shape {shape} and chunks {chunks}")

        # initialize new ROI table
        bbox_dataframe_list = []

    ##############
    # Filter nuclei by parent mask ###
    ##############

    # nuclei are masked by parent object (e.g. organoid) to only select nuclei belonging to parent.
    # load well image as dask array for parent objects
    r0_dask_parent = da.from_zarr(f"{zarr_url}/labels/{label_name_obj}/{level}")

    # Read Zarr metadata
    r0_ngffmeta_parent = load_NgffImageMeta(f"{zarr_url}/labels/{label_name_obj}")
    r0_xycoars_parent = r0_ngffmeta_parent.coarsening_xy
    r0_pixmeta_parent = r0_ngffmeta_parent.get_pixel_sizes_zyx(level=level)

    r0_idlist_parent = convert_ROI_table_to_indices(
        r0_adata,
        level=level,
        coarsening_xy=r0_xycoars_parent,
        full_res_pxl_sizes_zyx=r0_pixmeta_parent,
    )

    check_valid_ROI_indices(r0_idlist_parent, roi_table)

    r0_labels = r0_adata.obs_vector("label")
    # initialize variables
    compute = True  # convert to numpy array from dask

    # for each parent object (e.g. organoid) in r0...
    sphericity_flag = 0
    object_count = 0
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

        seg = mask_by_parent_object(
            seg, r0_dask_parent, r0_idlist_parent, row_int, r0_org_label
        )

        ##############
        # Perform label fusion and edge detection  ###
        ##############

        # calculate mean volume of nuclei or cells in object
        size_mean = calculate_mean_volume(seg)

        # expand labels in each z-slice and combine together
        seg_fill = np.zeros_like(seg)

        # the number of pixels by which to expand is a function of average nuclear size within the organoid.
        expandby_pix = int(round(expandby_factor * equivalent_diam(size_mean)))
        if expandby_pix == 0:
            logger.warning(
                "Equivalent diameter is 0 or negative, thus labels not expanded. Check segmentation quality"
            )

        iterations = int(round(expandby_pix / 2))

        # loop over each zslice and expand the labels, then fill holes
        for i, zslice in enumerate(seg):
            zslice = expand_labels(
                zslice, expandby_pix
            )  # expand labels in each zslice in xy
            zslice = binary_fill_holes(zslice)  # fill holes
            # to revert the expansion, erode by half of the expanded pixels ...
            # ...(since disk(1) has a radius of 1, i.e. diameter of 2)
            seg_fill[i, :, :] = binary_erosion(
                zslice, disk(1), iterations=iterations
            )  # erode down to original size

        # 3D gaussian blur
        seg_fill_8bit = (seg_fill * 255).astype(np.uint8)

        # calculate sigma based on z,y,x pixel spacing metadata, so that sigma scales with anisotropy
        pixel_anisotropy = r0_pixmeta[0] / np.array(
            r0_pixmeta
        )  # (z, y, x) where z is normalized to 1, e.g. (1, 3, 3)
        sigma = tuple(sigma_factor * x for x in pixel_anisotropy)
        blurred = gaussian(seg_fill_8bit, sigma=sigma, preserve_range=False)

        # Canny filter to detect gaussian edges
        edges_canny = np.zeros_like(blurred)
        # threshold
        blurred[blurred < canny_threshold] = 0  # was 0.15, 0.3

        padded_zslice_count = 0
        for i, zslice in enumerate(blurred):
            padded = False
            if np.count_nonzero(zslice) == 0:  # if all values are 0, skip this zslice
                continue
            else:
                image_border = load_border_values(zslice)
                # Pad z-slice border with 0 so that Canny filter generates a closed surface when there are non-zero
                # values at edge
                if np.any(image_border):
                    zslice = np.pad(zslice, 1)
                    padded_zslice_count += 1
                    padded = True

                edges = canny(zslice)
                edges = binary_fill_holes(edges)
                if padded:
                    edges = remove_border(edges)
                edges_canny[i, :, :] = edges

        edges_canny = (edges_canny * 255).astype(np.uint8)

        edges_canny = label(remove_small_objects(edges_canny, int(expandby_pix / 2)))

        # Check whether new label map has a single value, otherwise result is discarded and object skipped
        maxvalue = np.amax(edges_canny)
        if maxvalue != 1:
            if maxvalue == 0:
                logger.warning(
                    f"No 3D label and mesh saved for object {r0_org_label}. "
                    f"Result of canny edge detection is empty"
                )
                continue
            else:  # for max values greater than 1 or less than 0
                logger.warning(
                    f"No 3D label and mesh saved for object {r0_org_label}. Detected {maxvalue} labels. "
                    f"Is the shape composed of {maxvalue} distinct objects?"
                )
                continue

        logger.info(
            f"Successfully calculated 3D label map for object label {r0_org_label} using parameters:"
            f"\n\texpanded by {expandby_pix} pix, \n\teroded by {iterations*2} pix, "
            f"\n\tgaussian blurred with sigma = {np.round(sigma,1)}"
        )

        if padded_zslice_count > 0:
            logger.info(
                f"Object {r0_org_label} has non-zero pixels touching image border. Image processing "
                f"completed successfully, however consider reducing sigma_factor "
                f"or increasing the canny_threshold to reduce risk of cropping shape edges."
            )
        object_count += 1
        ##############
        # Calculate and save mesh  ###
        ##############

        if calculate_mesh:
            # Make mesh with vtkDiscreteFlyingEdges3D algorithm
            # Set spacing to ome-zarr pixel spacing metadata. Mesh will be in physical units (um)
            spacing = tuple(np.array(r0_pixmeta))  # z,y,x e.g. (0.6, 0.216, 0.216)

            # Pad border with 0 so that the mesh forms a manifold
            edges_canny_padded = np.pad(edges_canny, 1)
            mesh_polydata_organoid = labels_to_mesh(
                edges_canny_padded,
                spacing,
                polynomial_degree=polynomial_degree,
                pass_band_param=passband,
                feature_angle=feature_angle,
                target_reduction=target_reduction,
                smoothing_iterations=smoothing_iterations,
                margin=5,
                show_progress=False,
            )
            # Save mesh
            save_transform_path = (
                f"{zarr_url}/meshes/{label_name_obj}_from_{label_name}"
            )
            os.makedirs(save_transform_path, exist_ok=True)
            # Save name is the organoid label id
            save_name = f"{int(r0_org_label)}.stl"
            export_stl_polydata(
                os.path.join(save_transform_path, save_name), mesh_polydata_organoid
            )

            logger.info(
                f"Successfully generated surface mesh for object label {r0_org_label}"
            )

            volume, surface_area = get_mass_properties(mesh_polydata_organoid)
            sphr = mesh_sphericity(volume, surface_area)

            if sphr > 1.2:
                logger.warning(
                    f"Detected high sphericity = {np.round(sphr,3)} for object {r0_org_label}. "
                    f"Check mesh quality."
                )
                sphericity_flag += 1

        ##############
        # Save labels and make ROI table ###
        ##############

        if save_labels:
            # store labels as new label map in zarr
            # note that pixels of overlap in the case where two meshes are touching are overwritten by the last
            # written object
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
            new_label3d_dask = da.from_zarr(f"{zarr_url}/labels/{output_label_name}/0")
            # load region of current object from disk, will include any previously processed neighboring objects
            seg_ondisk = load_region(
                data_zyx=new_label3d_dask,
                region=region,
                compute=compute,
            )

            # check that dimensions of rois match
            if seg_ondisk.shape != edges_canny.shape:
                raise ValueError(
                    "Computed label image must match image dimensions of bounding box during saving"
                )

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

            # make new ROI table
            origin_zyx = convert_indices_to_origin_zyx(r0_idlist[row_int])

            bbox_df = array_to_bounding_box_table(
                edges_canny_label,
                r0_pixmeta,
                origin_zyx=origin_zyx,
            )

            bbox_dataframe_list.append(bbox_df)

            overlap_list = []
            for df in bbox_dataframe_list:
                overlap_list.extend(get_overlapping_pairs_3D(df, r0_pixmeta))
            if len(overlap_list) > 0:
                logger.warning(f"{len(overlap_list)} bounding-box pairs overlap")

    # Starting from on-disk highest-resolution data, build and write to disk a
    # pyramid of coarser levels
    if save_labels:

        build_pyramid(
            zarrurl=f"{zarr_url}/labels/{output_label_name}",
            overwrite=True,
            num_levels=r0_ngffmeta.num_levels,
            coarsening_xy=r0_ngffmeta.coarsening_xy,
            chunksize=chunks,
            aggregation_function=np.max,
        )

        logger.info(
            f"Built a pyramid for the {zarr_url}/labels/{output_label_name} label image"
        )

        bbox_table = format_roi_table(bbox_dataframe_list)
        # Write to zarr group
        logger.info(
            f"Writing new bounding-box ROI table to {zarr_url}/tables/{output_roi_table_name}"
        )

        table_attrs = {
            "type": "ngff:region_table",
            "region": {"path": f"../labels/{output_label_name}"},
            "instance_key": "label",
        }

        write_table(
            zarr.group(zarr_url),
            output_roi_table_name,
            bbox_table,
            overwrite=True,
            table_attrs=table_attrs,
        )

    if calculate_mesh:
        # Check how many objects out of well have a sphericity flag
        logger.info(
            f"{sphericity_flag} out of {object_count} meshed objects are flagged for high sphericity, "
            f"which can indicate a highly complex mesh surface."
        )
        if sphericity_flag > 0.1 * object_count:
            # if more than 10% of objects have a sphericity flag, raise warning
            logger.warning(
                "Detected high fraction of suspicious organoid meshes in well. Inspect mesh quality "
                "and tune task parameters for label expansion and mesh smoothing accordingly. "
            )

    logger.info(
        f"End surface_mesh_multiscale task for {zarr_url}/labels/{output_label_name}"
    )

    return {}


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=surface_mesh_multiscale,
        logger_name=logger.name,
    )
