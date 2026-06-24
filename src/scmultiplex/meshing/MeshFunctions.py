# Copyright (C) 2024 Friedrich Miescher Institute for Biomedical Research

##############################################################################
#                                                                            #
# Author: Nicole Repina              <nicole.repina@fmi.ch>                  #
# Author: Raphael Ortiz              <raphael.ortiz@fmi.ch>                  #
#                                                                            #
##############################################################################

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import vtk
from scipy.ndimage import find_objects
from scipy.spatial.distance import cdist
from tqdm import tqdm
from vtk import vtkPolyData
from vtkmodules.numpy_interface import dataset_adapter as dsa
from vtkmodules.util import numpy_support
from vtkmodules.vtkCommonCore import VTK_DOUBLE, vtkIdList
from vtkmodules.vtkFiltersCore import vtkFeatureEdges, vtkIdFilter

logger = logging.getLogger("annotate_child_mesh")


def numpy_img_to_vtk(img, spacing, origin=(0.0, 0.0, 0.0), deep_copy=True):
    """Converts a numpy array to vtk image data.

    Args:
        img: numpy array
        spacing: tuple defining the px/voxel size
        origin: origin point in physical coordinates
        deep_copy: if False memory will be shared with the original numpy array.
        It requires keeping a handle on the numpy array to prevent garbage collection.
    """

    vtk_data = numpy_support.numpy_to_vtk(
        num_array=img.ravel(order="C"), deep=deep_copy
    )
    imageVTK = vtk.vtkImageData()
    imageVTK.SetSpacing(spacing[::-1])
    imageVTK.SetOrigin(origin[::-1])
    imageVTK.SetDimensions(img.shape[::-1])
    imageVTK.GetPointData().SetScalars(vtk_data)

    return imageVTK


def smooth_me(
    vtkalgorithmoutput,
    smoothing_iterations,
    pass_band_param,
    feature_angle,
    target_reduction,
):
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputConnection(vtkalgorithmoutput)
    smoother.SetNumberOfIterations(
        smoothing_iterations
    )  # this has little effect on the error!
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOn()
    smoother.SetFeatureAngle(feature_angle)
    smoother.SetPassBand(pass_band_param)  # from 0 to 2, 2 keeps high frequencies
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.GenerateErrorScalarsOff()
    smoother.GenerateErrorVectorsOff()
    smoother.Update()

    # vtkQuadricDecimation looks cleaner than vtkDecimatePro (no unexpected sharp edges)
    # but drop points scalar value --> can be added back if doing one instance a time
    decimate = vtk.vtkQuadricDecimation()
    decimate.SetInputConnection(smoother.GetOutputPort())
    decimate.SetTargetReduction(target_reduction)
    decimate.VolumePreservationOn()
    decimate.Update()
    return decimate


def extract_smooth_mesh(
    imageVTK,
    label_range,
    polynomial_degree=30,
    pass_band_param=0.01,
    feature_angle=160,
    target_reduction=0.98,
    smoothing_iterations=1,
):
    """Extract mesh/contour for labels in imageVTK, smooth and decimate.

    Multiple labels can be extracted at once, however touching labels
    will share vertices and the label ids are lost during smoothing/decimation.
    Processing is slow for small objects in a large volume and should be cropped beforehand.

    Args:
        imageVTK: vtk image data
        label_range: range of labels to extract. A tuple (l,l) will extract
            a mesh for a single label id l
        polynomial_degree: number of iterations for vtkWindowedSincPolyDataFilter
        pass_band_param: pass band param in range [0.,2.] for vtkWindowedSincPolyDataFilter.
            Lower value remove higher frequencies.
        feature_angle: feature angle for sharp edge identification used
            for vtk FeatureEdgeSmoothing
        target_reduction: target reduction for vtkQuadricDecimation
        smoothing_iterations: the number of iterations that mesh smoothing and decimation is run
    """
    n_contours = label_range[1] - label_range[0] + 1

    # alternative vtkDiscreteMarchingCubes is slower and creates some wierd missalignment lines when applied to
    # tight crops
    dfe = vtk.vtkDiscreteFlyingEdges3D()
    dfe.SetInputData(imageVTK)
    dfe.ComputeScalarsOff()  # numpy image labels --> cells (faces) scalar values
    dfe.ComputeNormalsOff()
    dfe.ComputeGradientsOff()
    dfe.InterpolateAttributesOff()
    dfe.GenerateValues(
        n_contours, label_range[0], label_range[1]
    )  # numContours, rangeStart, rangeEnd
    dfe.Update()

    algorithmoutput = dfe.GetOutputPort()

    for n in range(smoothing_iterations):
        # after first iteration, use a fixed target reduction that almost entirely preserves number of triangles (0.1)
        # scale pass_band_param and feature_angle with iteration to increase smoothing
        tr = target_reduction
        pbp = pass_band_param
        fa = feature_angle

        if n > 0:
            tr = 0.1
            pbp = pbp / (2**n)
            fa = fa + (5 * n)

        # cap feature_angle at 180 degrees
        if fa > 180:
            fa = 180

        decimated = smooth_me(algorithmoutput, polynomial_degree, pbp, fa, tr)

        if n == (smoothing_iterations - 1):
            output = decimated.GetOutput()
            return output  # return smoothened vtk.vtkPolyData
        else:
            algorithmoutput = decimated.GetOutputPort()
            continue
    raise RuntimeError("Mesh extraction failed: smoothing loop did not return output.")


def labels_to_mesh(
    labels,
    spacing,
    polynomial_degree,
    pass_band_param,
    feature_angle,
    target_reduction,
    smoothing_iterations,
    margin,
    show_progress=False,
):
    """Extract mesh/contour for a labels provided as a numpy array, smooth and decimate.

    Meshes are extracted one label at a time one object crop.

    Args:
        labels: numpy array of 3D segmentation
        spacing: tuple defining the px/voxel size
        polynomial_degree: number of iterations for vtkWindowedSincPolyDataFilter
        smoothing_iterations: number of iterations for vtkWindowedSincPolyDataFilter
        pass_band_param: pass band param in range [0.,2.] for vtkWindowedSincPolyDataFilter.
            Lower value remove higher frequencies.
        feature_angle: feature angle for sharp edge identification used
            for vtk FeatureEdgeSmoothing
        target_reduction: target reduction for vtkQuadricDecimation
        smoothing_iterations: the number of iterations that mesh smoothing and decimation is run
        margin: margin bounding box used to crop each label. Needs at least margin=1 to extract
        closed contours (i.e. label should not touch the bounding box), slightly more for the
        smoothing operation.
        show_progress: display a progress bar. Useful for images containing many labels
    """

    appendFilter = vtk.vtkAppendPolyData()

    iterable = find_objects(labels)
    if show_progress:
        iterable = tqdm(iterable)

    for idx, loc in enumerate(iterable, start=1):
        if loc:
            loc = tuple(
                slice(max(0, sl.start - margin), sl.stop + margin) for sl in loc
            )
            crop = (labels[loc] == idx).astype(np.uint8)
            # TODO: here, add func that calculates origin of parent object.
            # pass this in as argument to offset the nuclear origin by this amount
            # add origin_offset argument
            # this ensures that nuclei match center coordinate of parent
            origin = tuple(sl.start * s for sl, s in zip(loc, spacing))
            imageVTK = numpy_img_to_vtk(crop, spacing, origin, deep_copy=False)
            instance_mesh = extract_smooth_mesh(
                imageVTK,
                (1, 1),
                polynomial_degree,
                pass_band_param,
                feature_angle,
                target_reduction,
                smoothing_iterations,
            )

            cleaner = vtk.vtkCleanPolyData()
            cleaner.SetInputData(instance_mesh)
            cleaner.Update()
            instance_mesh_cleaned = cleaner.GetOutput()

            # add the label id as point data
            scalars = numpy_support.numpy_to_vtk(
                num_array=np.ones(instance_mesh_cleaned.GetNumberOfPoints()) * idx,
                deep=True,
                array_type=vtk.VTK_INT,
            )
            scalars.SetName("label_id")
            instance_mesh_cleaned.GetPointData().SetScalars(scalars)

            appendFilter.AddInputData(instance_mesh_cleaned)

    appendFilter.Update()

    return appendFilter.GetOutput()


def upsample_mesh(
    mesh: vtk.vtkPolyData,
    target_points: int,
) -> vtk.vtkPolyData:
    num_points = mesh.GetNumberOfPoints()

    # Subdivide until exceed the target point count
    subdivisions = 0
    estimated = num_points
    while estimated < target_points:
        estimated *= 2
        subdivisions += 1

    subdivide = vtk.vtkLoopSubdivisionFilter()
    subdivide.SetNumberOfSubdivisions(subdivisions)
    subdivide.SetInputData(mesh)
    subdivide.Update()

    dense_mesh = subdivide.GetOutput()

    # if overshoot with upsampling, decimate
    mesh = downsample_mesh(dense_mesh, target_points)

    return mesh


def downsample_mesh(
    mesh: vtk.vtkPolyData,
    target_points: int,
) -> vtk.vtkPolyData:
    current_points = mesh.GetNumberOfPoints()
    # target_reduction = 0.75 will try to reduce the triangle count by 75%.
    target_reduction = 1.0 - (target_points / current_points)

    # note that vtkQuadraticDecimation() gives smoother surface results than vtkDecimatePro()
    # vtkDecimatePro() introduces sharp angles on surface
    decimate = vtk.vtkQuadricDecimation()
    decimate.SetInputData(mesh)
    # restrict reduction to range 0 (no reduction) - 0.9 (90% reduction)
    #  not based on point count directly
    decimate.SetTargetReduction(min(max(target_reduction, 0.0), 0.9))
    # helps retain the overall shape and silhouette of the object
    decimate.VolumePreservationOn()
    # enable the use of attribute-aware decimation, which considers scalar
    # data (e.g., color, temperature, labels) if present
    decimate.AttributeErrorMetricOn()
    decimate.Update()

    return decimate.GetOutput()


def add_mesh_points_attribute(mesh, attribute_name, label_mapping):
    """Adds points attribute to vtk polydata.

    Args:
        mesh: vtk polydata
        attribute_name: name of the attribute to be added
        label_mapping: dict mapping label ids (vtk PointData) to the new attribute (ScalarArray).

    """
    # mesh: polydata

    # turn labels_mapping into numpy look up table
    lut_dtype = type(next(iter(label_mapping.values())))
    lut = np.zeros(max(label_mapping.keys()) + 1, dtype=lut_dtype)
    for key, val in label_mapping.items():
        lut[key] = val

    point_labels = numpy_support.vtk_to_numpy(mesh.GetPointData().GetAttribute(0))
    attributes_array = lut[point_labels]
    scalars = numpy_support.numpy_to_vtk(attributes_array, deep=True)
    scalars.SetName(attribute_name)
    mesh.GetPointData().AddArray(scalars)


def get_convex_hull(polydata):
    """Get convex hull of input mesh using vtkHull()"""
    hull = vtk.vtkHull()
    hull.SetInputData(polydata)
    hull.AddCubeFacePlanes()  # generate bounding box
    hull.AddRecursiveSpherePlanes(5)
    hull.Update()
    clean = vtk.vtkStaticCleanPolyData()
    clean.SetInputConnection(hull.GetOutputPort())
    clean.Update()
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(clean.GetOutputPort())
    tri.Update()
    return tri.GetOutput()


def get_bounding_box(polydata):
    """Get convex hull of input mesh using vtkHull()"""
    hull = vtk.vtkHull()
    hull.SetInputData(polydata)
    hull.AddCubeFacePlanes()  # generate bounding box
    hull.Update()
    clean = vtk.vtkStaticCleanPolyData()
    clean.SetInputConnection(hull.GetOutputPort())
    clean.Update()
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(clean.GetOutputPort())
    tri.Update()

    return tri.GetOutput()


def get_mass_properties(polydata):
    """
    Get volume and surface area of input vtk polydata mesh object.
    """
    massProperties = vtk.vtkMassProperties()
    massProperties.SetInputData(polydata)

    volume = massProperties.GetVolume()
    surface_area = massProperties.GetSurfaceArea()

    return volume, surface_area


def get_centroid(polydata):
    """
    Get centroid of input vtk polydata mesh object.
    """
    coords = numpy_support.vtk_to_numpy(polydata.GetPoints().GetData())
    centroid = coords.mean(axis=0, keepdims=True)
    return centroid


def get_max_length(polydata):
    coords = numpy_support.vtk_to_numpy(polydata.GetPoints().GetData())
    hdist = cdist(coords, coords, metric="euclidean")

    # Get the farthest apart points and their distance
    indeces = np.unravel_index(hdist.argmax(), hdist.shape)
    most_distant_point_pair = [coords[indeces[0]], coords[indeces[1]]]
    maxdist = hdist[indeces]

    return maxdist, most_distant_point_pair


# adjust_edge_curvatures is from https://examples.vtk.org/site/Python/PolyData/CurvaturesAdjustEdges/
def adjust_edge_curvatures(source, curvature_name, epsilon=1.0e-08):
    """
    This function adjusts curvatures along the edges of the surface by replacing
     the value with the average value of the curvatures of points in the neighborhood.

    Remember to update the vtkCurvatures object before calling this.

    :param source: A vtkPolyData object corresponding to the vtkCurvatures object.
    :param curvature_name: The name of the curvature, 'Gauss_Curvature' or 'Mean_Curvature'.
    :param epsilon: Absolute curvature values less than this will be set to zero.
    :return:
    """

    def point_neighbourhood(pt_id):
        """
        Find the ids of the neighbours of pt_id.

        :param pt_id: The point id.
        :return: The neighbour ids.
        """
        """
        Extract the topological neighbors for point pId. In two steps:
        1) source.GetPointCells(pt_id, cell_ids)
        2) source.GetCellPoints(cell_id, cell_point_ids) for all cell_id in cell_ids
        """
        cell_ids = vtkIdList()
        source.GetPointCells(pt_id, cell_ids)
        neighbour = set()
        for cell_idx in range(0, cell_ids.GetNumberOfIds()):
            cell_id = cell_ids.GetId(cell_idx)
            cell_point_ids = vtkIdList()
            source.GetCellPoints(cell_id, cell_point_ids)
            for cell_pt_idx in range(0, cell_point_ids.GetNumberOfIds()):
                neighbour.add(cell_point_ids.GetId(cell_pt_idx))
        return neighbour

    def compute_distance(pt_id_a, pt_id_b):
        """
        Compute the distance between two points given their ids.

        :param pt_id_a:
        :param pt_id_b:
        :return:
        """
        pt_a = np.array(source.GetPoint(pt_id_a))
        pt_b = np.array(source.GetPoint(pt_id_b))
        return np.linalg.norm(pt_a - pt_b)

    # Get the active scalars
    source.GetPointData().SetActiveScalars(curvature_name)
    np_source = dsa.WrapDataObject(source)
    curvatures = np_source.PointData[curvature_name]

    #  Get the boundary point IDs.
    array_name = "ids"
    id_filter = vtkIdFilter()
    id_filter.SetInputData(source)
    id_filter.SetPointIds(True)
    id_filter.SetCellIds(False)
    id_filter.SetPointIdsArrayName(array_name)
    id_filter.SetCellIdsArrayName(array_name)
    id_filter.Update()

    edges = vtkFeatureEdges()
    edges.SetInputConnection(id_filter.GetOutputPort())
    edges.BoundaryEdgesOn()
    edges.ManifoldEdgesOff()
    edges.NonManifoldEdgesOff()
    edges.FeatureEdgesOff()
    edges.Update()

    edge_array = edges.GetOutput().GetPointData().GetArray(array_name)
    boundary_ids = []
    for i in range(edges.GetOutput().GetNumberOfPoints()):
        boundary_ids.append(edge_array.GetValue(i))
    # Remove duplicate Ids.
    p_ids_set = set(boundary_ids)

    # Iterate over the edge points and compute the curvature as the weighted
    # average of the neighbours.
    count_invalid = 0
    for p_id in boundary_ids:
        p_ids_neighbors = point_neighbourhood(p_id)
        # Keep only interior points.
        p_ids_neighbors -= p_ids_set
        # Compute distances and extract curvature values.
        curvs = [curvatures[p_id_n] for p_id_n in p_ids_neighbors]
        dists = [compute_distance(p_id_n, p_id) for p_id_n in p_ids_neighbors]
        curvs = np.array(curvs)
        dists = np.array(dists)
        curvs = curvs[dists > 0]
        dists = dists[dists > 0]
        if len(curvs) > 0:
            weights = 1 / np.array(dists)
            weights /= weights.sum()
            new_curv = np.dot(curvs, weights)
        else:
            # Corner case.
            count_invalid += 1
            # Assuming the curvature of the point is planar.
            new_curv = 0.0
        # Set the new curvature value.
        curvatures[p_id] = new_curv

    #  Set small values to zero.
    if epsilon != 0.0:
        curvatures = np.where(abs(curvatures) < epsilon, 0, curvatures)
        # Curvatures is now an ndarray
        curv = numpy_support.numpy_to_vtk(
            num_array=curvatures.ravel(), deep=True, array_type=VTK_DOUBLE
        )
        curv.SetName(curvature_name)
        source.GetPointData().RemoveArray(curvature_name)
        source.GetPointData().AddArray(curv)
        source.GetPointData().SetActiveScalars(curvature_name)
    return


# TODO: generalize to also calculate 'Mean_Curvature'.
def get_gaussian_curvatures(polydata, curvature_type="Gauss_Curvature"):
    """
    Calculate Gaussian Curvature for a 3D mesh. See also https://vtk.org/doc/nightly/html/classvtkCurvatures.html

    :param polydata: Input vtkPolyData object
    :param curvature_type: The name of the curvature, 'Gauss_Curvature'
    :return:
    :polydata: vtkPolyData object with calculated curvature set as active scalar
    :scalar_range: (min, max) value of calculated curvature
    :curvatures_numpy: curvature values as numpy array
    """
    curvatures = vtk.vtkCurvatures()
    curvatures.SetInputData(polydata)
    curvatures.SetCurvatureTypeToGaussian()
    curvatures.Update()

    adjust_edge_curvatures(curvatures.GetOutput(), curvature_type)
    polydata.GetPointData().AddArray(
        curvatures.GetOutput().GetPointData().GetAbstractArray(curvature_type)
    )
    scalar_range = polydata.GetPointData().GetScalars(curvature_type).GetRange()

    polydata.GetPointData().SetActiveScalars(
        curvature_type
    )  # visualize curvature when load into viewer
    np_source = dsa.WrapDataObject(polydata)
    curvatures_numpy = np_source.PointData[curvature_type]

    return polydata, scalar_range, curvatures_numpy


def get_curv_derivative(polydata):
    """
    Calculate derivative of curvature using vtkCellDerivatives()

    :param polydata: Input vtkPolyData object with calculated curvature as scalar
        (e.g. polydata output of get_gaussian_curvatures)
    :return:
    :polydata: vtkPolyData object with calculated curvature derivative.
    """
    curvdiriv = vtk.vtkCellDerivatives()
    curvdiriv.SetInputData(polydata)
    curvdiriv.SetVectorModeToComputeGradient()
    curvdiriv.Update()

    return curvdiriv.GetOutput()


def export_vtk_polydata(path, polydata):
    """Exports vtk polydata as *.vtp"""

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(path)
    writer.SetInputData(polydata)
    writer.SetCompressorTypeToZLib()
    writer.SetCompressionLevel(9)
    writer.Write()


def export_stl_polydata(path, polydata):
    """Exports vtk polydata as *.stl"""

    writer = vtk.vtkSTLWriter()
    writer.SetFileName(path)
    writer.SetInputData(polydata)
    writer.Write()


def read_stl_polydata(fpath):
    """Load .stl mesh from disk and return as vtkPolyData object"""
    reader = vtk.vtkSTLReader()
    reader.SetFileName(fpath)
    reader.Update()
    polydata = reader.GetOutput()
    return polydata


def read_vtp_polydata(fpath):
    """Load .vtp mesh from disk and return as vtkPolyData object"""
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(fpath)
    reader.Update()
    polydata = reader.GetOutput()
    return polydata


def load_mesh_as_polydata(
    mesh_dir: Union[str, Path],
    label_string: str,
) -> Optional[vtkPolyData]:
    """
    Load a mesh file for a given label and return it as polydata.

    The function searches for a mesh file with supported extensions
    (in order of preference: .stl, .vtk, .vtp) in the specified directory.
    The first matching file found is loaded and returned as polydata.

    Parameters
    ----------
    mesh_dir : str or pathlib.Path
        Directory containing mesh files.
    label_string : str
        Base filename (without extension) identifying the mesh.

    Returns
    -------
    polydata : vtkPolyData or None
        Loaded polydata object if a mesh file is found; otherwise None.
    """

    mesh_dir = Path(mesh_dir)

    mesh_extensions = [".stl", ".vtk", ".vtp"]  # order of preference
    for ext in mesh_extensions:
        mesh_path = mesh_dir / f"{label_string}{ext}"
        if mesh_path.is_file():
            if ext == ".stl":
                return read_stl_polydata(mesh_path)
            else:
                return read_vtp_polydata(mesh_path)

    return None


def filter_polydata_by_label_ids(
    polydata: vtk.vtkPolyData,
    keep_label_ids: list[int] | np.ndarray,
    label_array_name: str = "label_id",
    logger: logging.Logger | None = None,
) -> vtk.vtkPolyData:
    """
    Remove objects from a vtkPolyData mesh based on their label IDs.

    Parameters
    ----------
    polydata : vtk.vtkPolyData
        Input mesh. Must contain a point-data array with label IDs.
    keep_label_ids : list[int] | np.ndarray
        Label IDs to keep in the output mesh.
        Any object whose label_id is not in this list is removed.
    label_array_name : str
        Name of the point-data array containing label IDs.
    logger : logging.Logger, optional
        Logger used to report which label IDs were removed from the mesh.
        If None, no logging is performed.

    Returns
    -------
    vtk.vtkPolyData
        Filtered mesh containing only objects with label IDs in
        keep_label_ids.

    Notes
    -----
    Label removal is determined from labels actually present in the mesh,
    not from keep_label_ids. This allows reporting which child objects
    were removed during filtering.
    """

    # Get label ID array from the polydata point data
    label_array_vtk = polydata.GetPointData().GetArray(label_array_name)

    if label_array_vtk is None:
        raise ValueError(
            f"Point-data array '{label_array_name}' not found in polydata."
        )

    # Convert VTK label array to NumPy
    label_ids = numpy_support.vtk_to_numpy(label_array_vtk)

    # Record labels present in the original mesh
    original_label_ids = set(np.unique(label_ids))

    # For each point, True means keep it, False means remove it
    keep_mask = np.isin(label_ids, keep_label_ids)

    # Convert boolean mask to 0/1 because VTK thresholding uses numeric arrays
    keep_array_vtk = numpy_support.numpy_to_vtk(
        keep_mask.astype(np.uint8),
        deep=True,
        array_type=vtk.VTK_UNSIGNED_CHAR,
    )
    keep_array_vtk.SetName("_keep")

    # Temporarily attach the keep/remove mask to the mesh
    polydata.GetPointData().AddArray(keep_array_vtk)

    # Keep only mesh elements whose _keep value is 1
    threshold = vtk.vtkThreshold()
    threshold.SetInputData(polydata)
    threshold.SetInputArrayToProcess(
        0,
        0,
        0,
        vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS,
        "_keep",
    )
    threshold.SetLowerThreshold(1)
    threshold.SetUpperThreshold(1)
    threshold.Update()

    # vtkThreshold outputs vtkUnstructuredGrid, so convert back to vtkPolyData
    geometry = vtk.vtkGeometryFilter()
    geometry.SetInputData(threshold.GetOutput())
    geometry.Update()

    filtered_polydata = geometry.GetOutput()

    # Remove temporary bookkeeping array
    filtered_polydata.GetPointData().RemoveArray("_keep")

    # Determine which labels were removed
    filtered_label_array_vtk = filtered_polydata.GetPointData().GetArray(
        label_array_name
    )

    if filtered_label_array_vtk is not None:
        filtered_label_ids = set(
            np.unique(numpy_support.vtk_to_numpy(filtered_label_array_vtk))
        )
    else:
        filtered_label_ids = set()

    removed_label_ids = sorted(original_label_ids - filtered_label_ids)

    if logger is not None and removed_label_ids:
        logger.info(
            f"...Removed {len(removed_label_ids)} child labels from mesh: "
            f"{removed_label_ids}"
        )

    return filtered_polydata


def get_label_ids_from_polydata(
    polydata: vtk.vtkPolyData,
    label_array_name: str = "label_id",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract label IDs from a polydata point-data array.

    Parameters
    ----------
    polydata : vtk.vtkPolyData
        Input mesh containing a point-data array with child object labels.
    label_array_name : str
        Name of the point-data array containing label IDs.

    Returns
    -------
    label_ids : np.ndarray
        Array of shape (n_points,) containing the label ID assigned
        to each mesh point.
    unique_label_ids : np.ndarray
        Unique label IDs present in the mesh.

    Raises
    ------
    ValueError
        If the requested label array does not exist in the polydata.
    """

    # Get label-id array from mesh point data
    label_id_vtk = polydata.GetPointData().GetArray(label_array_name)

    if label_id_vtk is None:
        raise ValueError(f"No '{label_array_name}' array found in point data.")

    # Convert VTK array to NumPy
    label_ids = numpy_support.vtk_to_numpy(label_id_vtk)

    # Get unique child labels present in the mesh
    unique_label_ids = np.unique(label_ids)

    return label_ids, unique_label_ids


def add_features_to_polydata(
    polydata: vtk.vtkPolyData,
    label_ids: np.ndarray,
    unique_label_ids: np.ndarray,
    feature_df: pd.DataFrame,
    annotate_by_features: list[str],
    column_prefix: str = "",
    logger: logging.Logger | None = None,
) -> None:
    """
    Add feature values from a DataFrame as point-data arrays on a mesh.

    Assumes that the DataFrame index contains child label IDs matching
    the mesh label_id values.

    Parameters
    ----------
    polydata : vtk.vtkPolyData
        Mesh to annotate.
    label_ids : np.ndarray
        Array of shape (n_points,) containing the child label assigned
        to each mesh point.
    unique_label_ids : np.ndarray
        Unique child labels present in the mesh.
    feature_df : pd.DataFrame
        DataFrame indexed by child label ID.
        Columns contain feature values to transfer to the mesh.
    annotate_by_features : list[str]
        List of feature columns to add as mesh point-data arrays.
    column_prefix : str
        Optional prefix added to the output array names. E.g.
        "" -> volume
        "r1." -> r1.volume
    logger : logging.Logger | None, optional
        Logger used to report warnings, such as missing feature columns.
        If None, warnings are suppressed.

    Raises
    ------
    ValueError
        If a mesh label exists but does not have a corresponding row
        in feature_df.
    """

    for col in annotate_by_features:

        # Skip missing feature columns
        if col not in feature_df.columns:
            if logger is not None:
                logger.warning(f"Feature column '{col}' not found. Skipping.")
            continue
        # Optional round-specific naming
        save_col_name = f"{column_prefix}{col}"

        # Create output array matching the source feature dtype
        new_array = np.zeros_like(
            label_ids,
            dtype=feature_df[col].dtype,
        )

        # Map:
        #     child_label_id -> feature_value
        #
        # Example:
        #     {1: 123.4, 2: 87.1, 3: 95.6}
        feat_map = feature_df[col].to_dict()

        # Assign feature value to every point belonging
        # to a given child label
        for label_id in unique_label_ids:

            if label_id not in feat_map:
                raise ValueError(
                    f"Child mesh label {label_id} " f"not found in feature table."
                )

            new_array[label_ids == label_id] = feat_map[label_id]

        # Convert NumPy array to VTK
        vtk_array = numpy_support.numpy_to_vtk(
            new_array,
            deep=True,
        )
        vtk_array.SetName(save_col_name)

        # Add annotation to mesh
        polydata.GetPointData().AddArray(vtk_array)
