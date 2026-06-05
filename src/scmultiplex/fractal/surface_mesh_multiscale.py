# Copyright (C) 2024 Friedrich Miescher Institute for Biomedical Research

##############################################################################
#                                                                            #
# Author: Nicole Repina              <nicole.repina@fmi.ch>                  #
#                                                                            #
##############################################################################


"""
Calculates 3D surface mesh of parent object (e.g. tissue, organoid)
from 3D cell-level segmentation of children (e.g. nuclei)
"""
import logging
from typing import Any, Optional, Union

import anndata as ad
import dask.array as da
import numpy as np
import zarr
from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.pyramids import build_pyramid
from fractal_tasks_core.roi import (
    check_valid_ROI_indices,
    convert_indices_to_regions,
    convert_ROI_table_to_indices,
    get_overlapping_pairs_3D,
    load_region,
)
from fractal_tasks_core.tables import write_table
from fractal_tasks_core.tasks.io_models import InitArgsRegistrationConsensus
from pydantic import validate_call

from scmultiplex.features.FeatureFunctions import mesh_sphericity
from scmultiplex.fractal.fractal_helper_functions import (
    clear_mesh_folder,
    compute_and_save_mesh,
    format_roi_table,
    get_zattrs,
    initialize_new_label,
    save_new_label_and_bbox_df,
)
from scmultiplex.meshing.FilterFunctions import mask_by_parent_object, remove_xy_pad
from scmultiplex.meshing.LabelFusionFunctions import (
    fill_holes_by_slice,
    filter_by_volume,
    run_label_fusion,
    select_largest_component,
)
from scmultiplex.meshing.MeshFunctions import get_mass_properties

logger = logging.getLogger(__name__)


@validate_call
def surface_mesh_multiscale(
    *,
    # Fractal arguments
    zarr_url: str,
    init_args: InitArgsRegistrationConsensus,
    # Task-specific arguments
    label_name: str = "nuc",
    group_by: Union[str, None] = None,
    roi_table: str = "org_ROI_table_linked",
    multiscale: bool = True,
    save_mesh: bool = True,
    new_mesh_name: Optional[str] = None,
    expandby_factor: float = 0.6,
    sigma_factor: float = 5,
    canny_threshold: float = 0.3,
    mask_contour_by_parent: bool = False,
    fill_holes: bool = False,
    filter_by_object_volume: bool = False,
    object_volume_filter_threshold: int = 60000,
    filter_children_by_volume: bool = False,
    child_volume_filter_threshold: float = 0.05,
    polynomial_degree: int = 30,
    passband: float = 0.01,
    feature_angle: int = 160,
    target_reduction: float = 0.98,
    smoothing_iterations: int = 1,
    resample_mesh_to_target_point_count: bool = False,
    target_point_count: int = 5000,
    overwrite_folder: bool = True,
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
        label_name: Label name of segmentation for which mesh is calculated. When Multiscale = True, this is the label
            name of child objects (e.g. nuclei) that will be used for multiscale surface estimation.
        group_by: Label name of segmentated objects that are parents of label_name. If None (default), no grouping
            is applied and meshes are calculated for the input object (label_name).
            Instead, if a group_by label is specified, the
            label_name objects will be masked and grouped by this object. For example, when group_by = 'org', the
            nuclear segmentation is masked by the organoid parent and all nuclei belonging to the parent are loaded
            as a label image. Thus, when Multiscale = False, the calculated mesh contains multiple child objects. When
            Multiscale = True, a new labelmap is generated as a result of child fusion to generate the
            3D parent (organoid-level) shape.
        roi_table: Name of the ROI table used to iterate over objects and load object regions. If group_by = None, this
            is the ROI table that corresponds to the label_name objects. If group_by is passed, this is the ROI table
            for the group_by objects, e.g. org_ROI_table.
        multiscale: if True, a new labelmap is generated as a result of child fusion to generate the
            3D parent (organoid-level) shape. This label output is called {group_by}_from_{label_name},
            with corresponding ROI table name {group_by}_ROI_table_from_{label_name}. This label image is
            optionally saved as a mesh if save_mesh = True. If Multiscale = False, no multiscale label map computation
            is performed and a smoothened mesh of the input label_name is generated.
        save_mesh: if True, calculates and saves mesh on disk in the 'meshes' folder within zarr structure. Meshes
            saved as '.stl', except for the case of multi-object meshed (e.g. multiple nuclei within a parent organoid)
            that are saved as '.vtp' to preserve label ID information. Filename corresponds to parent object label id,
            or to label id in the case when group_by = None.
        new_mesh_name: Optionally new name for new label map (if multiscale) and mesh subfolder (if save_mesh) that is
            generated. If left None, default is {label_name}.
        expandby_factor: only used if Multiscale = True. Multiplier that specifies pixels by which to expand each
            nuclear mask for merging, float in range [0, 1 or higher], e.g. 0.2 means that 20% of mean of
            nuclear equivalent diameter is used.
        sigma_factor: only used if Multiscale = True. Float that specifies sigma (standard deviation, in pixels)
            for Gaussian kernel used for blurring to smoothen label image prior to edge detection.
            Higher values correspond to more blurring. Recommended range 5-15.
        canny_threshold: only used if Multiscale = True. Image values below this threshold are set to 0 after
            Gaussian blur. float in range [0,1]. Higher values result in tighter fit of mesh to nuclear surface.
        mask_contour_by_parent: if True, the final multiscale edges are masking by 2D parent object mask. Can be used
            to define cleaner edge borders between touching organoids, but may crop surface mask if higher
            blurring is desired.
        fill_holes: if True, the label image just prior to meshing has holes filled by iterating
            over slices. Useful for filling lumens in segmentation.
        filter_by_object_volume: if True, the label image is filtered by volume. This skips objects with lower
            volume (number of pixels, calculated after all processing and hole filling, if applied)
            than the object_volume_filter_threshold.
        object_volume_filter_threshold: Integer threshold for object volume filtering. Number of pixels. E.g. if
            set to 600, objects with a pixel count less than 600 are skipped. Only used if filter_by_object_volume
            is True.
        filter_children_by_volume: if True, performing volume filtering of nuclei to remove objects smaller
            than specified volume_filter_threshold.
        child_volume_filter_threshold: Multiplier that specifies cutoff for volumes below which nuclei are filtered out,
            float in range [0,1], e.g. 0.05 means that 5% of median of nuclear volume distribution is used as cutoff.
            Specify this value if volume filtering is desired. Default 0.05.
        polynomial_degree: Mesh smoothing parameter. The number of polynomial degrees during surface mesh smoothing with
            vtkWindowedSincPolyDataFilter determines the maximum number of smoothing passes.
            This number corresponds to the degree of the polynomial that is used to approximate the windowed sinc
            function. Usually 10-20 iteration are sufficient. Higher values have little effect on smoothing.
            For further details see VTK vtkWindowedSincPolyDataFilter documentation.
        passband: Mesh smoothing parameter. Float in range [0,2] that specifies the PassBand for the windowed sinc
            filter in vtkWindowedSincPolyDataFilter during mesh smoothing. Lower passband values produce more
            smoothing, due to filtering of higher frequencies. For further details see
            VTK vtkWindowedSincPolyDataFilter documentation.
        feature_angle: Mesh smoothing parameter. Integer in range [0,180] that specifies the feature angle for sharp
            edge identification used for vtk FeatureEdgeSmoothing. Higher values result in more smoothened
            edges in mesh. For further details see VTK vtkWindowedSincPolyDataFilter documentation.
        target_reduction: Mesh decimation parameter. Float in range [0,1].
            Target reduction is used during mesh decimation via
            vtkQuadricDecimation to reduce the number of triangles in a triangle mesh, forming a good approximation to
            the original geometry. Values closer to 1 indicate larger reduction and smaller mesh file size. Note that
            target_reduction is expressed as the fraction of the original number of triangles in mesh and so is
            proportional to original mesh size. Note the actual reduction may be less depending on triangulation and
            topological constraints. For further details see VTK vtkQuadricDecimation documentation.
        smoothing_iterations:  Mesh smoothing parameter.
            The number of iterations that mesh smoothing and decimation is run.
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
        resample_mesh_to_target_point_count: If True, the mesh is resampled to the target_point_count. All smoothing
            and decimation is performed as specified above. The smoothened mesh is then upsampled
            (with Loop Subdivision and Quadric Decimation) or downsampled (with Quadric Decimation), depending on
            whether the number of points in the smoothened mesh is lower or greater than the target_point_count,
            respectively. If True, all output meshes will have the same total number of points and triangles per
            object, though different point densities since object volumes are different. All .stl files also have
            same size. Note that depending on whether input mesh has an even or odd number of points, resampled output
            might still have slightly varying point/triangle counts between objects. If False, the original
            smoothened mesh is returned where the number of points differs between objects (but density per surface
            area is roughly the same).
        target_point_count: Integer number of points desired in remeshed object. Note that this is approximate and
            resampled output might still have slightly varying point/triangle counts between objects (e.g.
            depending on whether input mesh has an even or odd number of points).
            Only used if resample_mesh_to_target_point_count is True.
        overwrite_folder: If True, the output mesh folder, if exists, is cleared. Recommended to set True so
            that any meshes that already exist in folder but in current run are not processed are removed.


    """
    logger.info(
        f"Running for {zarr_url=}. \n"
        f"Calculating surface mesh per {roi_table=} for "
        f"{label_name=}."
    )

    if multiscale is True and group_by is None:
        raise ValueError(
            "Multiscale calculation is not possible without a provided group_by label for parent objects."
            " Check task inputs."
        )

    if group_by in [
        "",
        " ",
        "   ",
    ]:
        raise ValueError("Input group_by label name is not valid. Check task inputs.")

    # always use highest resolution label
    level = 0

    # Lazily load zarr array for reference cycle
    # load well image as dask array e.g. for nuclear segmentation
    label_dask = da.from_zarr(f"{zarr_url}/labels/{label_name}/{level}")

    # Read ROIs of objects
    roi_adata = ad.read_zarr(f"{zarr_url}/tables/{roi_table}")
    roi_attrs = get_zattrs(f"{zarr_url}/tables/{roi_table}")

    # Read Zarr metadata
    label_ngffmeta = load_NgffImageMeta(f"{zarr_url}/labels/{label_name}")
    label_xycoars = (
        label_ngffmeta.coarsening_xy
    )  # need to know when building new pyramids
    label_pixmeta = label_ngffmeta.get_pixel_sizes_zyx(level=level)

    # Create list of indices for 3D ROIs spanning the entire Z direction
    # Note that this ROI list is generated based on the input ROI table; if the input ROI table is for the group_by
    # objects, then label regions will be loaded based on the group_by ROIs
    roi_idlist = convert_ROI_table_to_indices(
        roi_adata,
        level=level,
        coarsening_xy=label_xycoars,
        full_res_pxl_sizes_zyx=label_pixmeta,
    )

    check_valid_ROI_indices(roi_idlist, roi_table)

    if len(roi_idlist) == 0:
        logger.warning("Well contains no objects")

    if new_mesh_name is None:
        label_root = label_name
    else:
        label_root = new_mesh_name

    if multiscale:
        # Initialize parameters to save the newly calculated label map
        # Save with same dimensions as child labels from which they are calculated

        output_label_name = f"{group_by}_from_{label_root}"
        output_roi_table_name = f"{output_label_name}_ROI_table"

        shape = label_dask.shape
        chunks = label_dask.chunksize

        new_label3d_array = initialize_new_label(
            zarr_url, shape, chunks, np.uint32, label_name, output_label_name, logger
        )

        logger.info(f"Mask will have shape {shape} and chunks {chunks}")

        # initialize new ROI table
        bbox_dataframe_list = []

    # Clear contents of mesh folder if it already exists
    if save_mesh and multiscale:
        # Mesh calculation for a new 3D object calculated from child objects (multiscale)
        mesh_folder_name = output_label_name
    elif save_mesh and group_by is not None and multiscale is False:
        # Mesh calculation for existing child objects that are part of the group_by object
        mesh_folder_name = f"{label_root}_grouped"
    elif save_mesh and group_by is None and multiscale is False:
        mesh_folder_name = f"{label_root}"

    if overwrite_folder:
        clear_mesh_folder(mesh_folder_name, zarr_url)

    ##############
    # Filter nuclei by parent mask ###
    ##############

    if group_by is not None:
        # Load group_by object segmentation to mask child objects by parent group_by object
        # Load well image as dask array for parent objects
        groupby_dask = da.from_zarr(f"{zarr_url}/labels/{group_by}/{level}")

        # Read Zarr metadata
        groupby_ngffmeta = load_NgffImageMeta(f"{zarr_url}/labels/{group_by}")
        groupby_xycoars = groupby_ngffmeta.coarsening_xy
        groupby_pixmeta = groupby_ngffmeta.get_pixel_sizes_zyx(level=level)

        groupby_idlist = convert_ROI_table_to_indices(
            roi_adata,
            level=level,
            coarsening_xy=groupby_xycoars,
            full_res_pxl_sizes_zyx=groupby_pixmeta,
        )

        check_valid_ROI_indices(groupby_idlist, roi_table)

    # Get labels to iterate over
    instance_key = roi_attrs["instance_key"]  # e.g. "label"

    # NGIO FIX, TEMP
    # Check that ROI_table.obs has the right column and extract label_value
    if instance_key not in roi_adata.obs.columns:
        if roi_adata.obs.index.name == instance_key:
            # Workaround for new ngio table
            roi_adata.obs[instance_key] = roi_adata.obs.index
        else:
            raise ValueError(
                f"In input ROI table, {instance_key=} "
                f" missing in {roi_adata.obs.columns=}"
            )

    roi_labels = roi_adata.obs_vector(instance_key)
    total_label_count = len(roi_labels)
    compute = True
    sphericity_flag = 0
    object_count = 0

    logger.info(
        f"Starting iteration over {total_label_count} detected objects in ROI table."
    )

    # For each object in input ROI table...
    for i, obsname in enumerate(roi_adata.obs_names):
        label_str = roi_labels[i]
        region = convert_indices_to_regions(roi_idlist[i])

        # Load label image of label_name object as numpy array
        seg = load_region(
            data_zyx=label_dask,
            region=region,
            compute=compute,
        )

        if group_by is not None:
            # Mask objects by parent group_by object
            seg, parent_mask = mask_by_parent_object(
                seg, groupby_dask, groupby_idlist, i, label_str
            )
            # Only proceed if labelmap is not empty
            if np.amax(seg) == 0:
                logger.warning(
                    f"Skipping object ID {label_str}. Label image contains no labeled objects."
                )
                # Skip this object
                continue
        else:
            # Check that label exists in object
            if float(label_str) not in seg:
                raise ValueError(
                    f"Object ID {label_str} does not exist in loaded segmentation image. Does input ROI "
                    f"table match label map?"
                )
            # Select label that corresponds to current object, set all other objects to 0
            seg[seg != float(label_str)] = 0

        ##############
        # Perform label fusion and edge detection  ###
        ##############
        if filter_children_by_volume:
            (
                seg,
                segids_toremove,
                removed_size_mean,
                size_mean,
                volume_cutoff,
            ) = filter_by_volume(seg, child_volume_filter_threshold)

            if len(segids_toremove) > 0:
                logger.info(
                    f"Volume filtering removed {len(segids_toremove)} cell(s) from object {label_str} "
                    f"that have a volume below the calculated {np.round(volume_cutoff,1)} pixel threshold"
                    f"\n Removed labels have a mean volume of {np.round(removed_size_mean,1)} and are the "
                    f"label id(s): "
                    f"\n {segids_toremove}"
                )

        if multiscale:
            xy_padwidth = int(sigma_factor)
            # Generate new 3D label image
            (
                edges_canny,
                expandby_pix,
                anisotropic_sigma,
                padded_zslice_count,
                roi_count,
            ) = run_label_fusion(
                seg,
                parent_mask,
                expandby_factor,
                sigma_factor,
                label_pixmeta,
                canny_threshold,
                xy_padwidth,
                mask_by_parent=mask_contour_by_parent,
            )

            # Perform checks
            if expandby_pix == 0:
                logger.warning(
                    "Equivalent diameter is 0 or negative, thus labels not expanded. "
                    "Check segmentation quality"
                )
                continue

            # Check whether is binary
            if np.amax(edges_canny) not in [0, 1]:
                raise ValueError("Edge detection results not binary.")

            if roi_count > 0:
                logger.info(
                    f"Successfully calculated 3D label map for object label {label_str}. Label expanded and eroded "
                    f"by {expandby_pix} pixels."
                )

                if roi_count > 1:
                    logger.info(
                        f"Object {label_str} contains more than 1 component. "
                        f"Largest component selected as label mask."
                    )
            else:
                logger.warning(
                    f"Empty result for object label  {label_str}. No label calculated. "
                    f"Is input region image empty?"
                )
                continue

            if padded_zslice_count > 0:
                logger.info(
                    f"Object {label_str} has non-zero pixels touching image border. Image processing "
                    f"completed successfully, however consider reducing sigma "
                    f"or increasing the canny_threshold to reduce risk of cropping shape edges."
                )

        ##############
        # Calculate and save mesh  ###
        ##############

        # TODO: redundant with mesh_folder_name lines at start of task, can simplify.
        if save_mesh and multiscale:
            # Mesh calculation for a new 3D object calculated from child objects (multiscale)
            label_image = edges_canny
            mesh_folder_name = output_label_name
            object_name = label_str
            sphericity_check = True
            save_as_stl = True
        elif save_mesh and group_by is not None and multiscale is False:
            # Mesh calculation for existing child objects that are part of the group_by object
            label_image = seg
            mesh_folder_name = f"{label_root}_grouped"
            object_name = label_str
            sphericity_check = False
            # Save as .vtp, since .stl does not support multiple object labels
            save_as_stl = False
        elif save_mesh and group_by is None and multiscale is False:
            label_image = seg
            mesh_folder_name = f"{label_root}"
            object_name = label_str
            sphericity_check = True
            save_as_stl = True
        else:
            logger.info(
                "Skipping mesh saving. If this is undesired, did you input "
                "correct labels and ROI tables?"
            )
            continue

        # Check that label image contains an object
        if np.amax(label_image) == 0:
            logger.warning(f"Label image is empty. Skipping object {label_str}!")
            continue

        # Select largest component, only for single-object meshes
        if group_by is None:
            label_image, roi_count = select_largest_component(label_image)
            if roi_count > 1:
                logger.warning(
                    f"Object {label_str} contains more than 1 component. "
                    f"Largest component selected as label mask."
                )

        # Fill holes, e.g. lumen
        if fill_holes and group_by is None:
            # fill holes in label image
            label_image = fill_holes_by_slice(label_image)

        # Filter out small volumes e.g. debris or segmentation mistakes
        if filter_by_object_volume:
            # get number of pixels in object
            counts = np.bincount(label_image.ravel())[-1]
            if counts < object_volume_filter_threshold:
                logger.warning(
                    f"Volume of object is less than threshold. Skipping object {label_str}!"
                )
                continue

        try:
            mesh_polydata = compute_and_save_mesh(
                label_image,
                label_str,
                label_pixmeta,
                polynomial_degree,
                passband,
                feature_angle,
                target_reduction,
                smoothing_iterations,
                zarr_url,
                mesh_folder_name,
                object_name,
                save_as_stl,
                resample_mesh_to_target_point_count,
                target_point_count,
            )
        except Exception as e:
            logger.warning(f"Failed to save mesh. Reason: {e}")
            continue

        object_count += 1

        if sphericity_check:
            volume, surface_area = get_mass_properties(mesh_polydata)
            sphr = mesh_sphericity(volume, surface_area)

            if sphr > 1.2:
                logger.warning(
                    f"Detected high sphericity = {np.round(sphr,3)} for object {label_str}. "
                    f"Check mesh quality."
                )
                sphericity_flag += 1

        ##############
        # Save labels and make ROI table ###
        ##############

        if multiscale:
            # Store labels as new label map in zarr
            # Note that pixels of overlap in the case where two meshes are touching are overwritten by the last
            # written object
            # Thus meshes are the most accurate representation of surface, labels may be cropped
            if not mask_contour_by_parent:
                # Remove pad
                edges_canny = remove_xy_pad(edges_canny, pad_width=xy_padwidth)

            # Convert edge detection label image value to match object label id
            edges_canny = edges_canny * int(label_str)

            bbox_df = save_new_label_and_bbox_df(
                edges_canny,
                new_label3d_array,
                zarr_url,
                output_label_name,
                region,
                label_pixmeta,
                compute,
                roi_idlist,
                i,
            )
            bbox_dataframe_list.append(bbox_df)

            overlap_list = []
            for df in bbox_dataframe_list:
                overlap_list.extend(get_overlapping_pairs_3D(df, label_pixmeta))

            if len(overlap_list) > 0:
                logger.warning(f"{len(overlap_list)} bounding-box pairs overlap")

    # Starting from on-disk highest-resolution data, build and write to disk a pyramid of coarser levels
    if multiscale:

        build_pyramid(
            zarrurl=f"{zarr_url}/labels/{output_label_name}",
            overwrite=True,
            num_levels=label_ngffmeta.num_levels,
            coarsening_xy=label_ngffmeta.coarsening_xy,
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

    if save_mesh and multiscale:
        # Check how many objects out of well have a sphericity flag
        logger.info(
            f"{sphericity_flag} out of {object_count} meshed objects are flagged for high sphericity, "
            "which can indicate a highly complex mesh surface."
        )
        if sphericity_flag > 0.1 * object_count:
            # if more than 10% of objects have a sphericity flag, raise warning
            logger.warning(
                "Detected high fraction of suspicious organoid meshes in well. Inspect mesh quality "
                "and tune task parameters for label expansion and mesh smoothing accordingly. "
            )
    if save_mesh:
        logger.info(f"Meshes saved in folder name {mesh_folder_name}")
    logger.info(
        f"Successfully processed {object_count} out of {total_label_count} labels."
    )
    logger.info(f"End surface_mesh_multiscale task for {zarr_url}/labels/{label_name}")

    return {}


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=surface_mesh_multiscale,
        logger_name=logger.name,
    )
