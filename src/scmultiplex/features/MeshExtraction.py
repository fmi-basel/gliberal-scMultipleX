# Copyright (C) 2024 Friedrich Miescher Institute for Biomedical Research

##############################################################################
#                                                                            #
# Author: Nicole Repina              <nicole.repina@fmi.ch>                  #
#                                                                            #
##############################################################################

from scmultiplex.features.FeatureFunctions import mesh_sphericity, mesh_extent, mesh_solidity, mesh_concavity, \
    mesh_asymmetry, mesh_aspect_ratio, mesh_surface_area_to_volume_norm
from scmultiplex.meshing.MeshFunctions import get_centroid, get_mass_properties, get_bounding_box, get_convex_hull


def get_mesh_measurements(polydata):

    ##############
    # Extract features  ###
    ##############

    # Generate hulls (convex hull and bounding box)
    convex_hull_polydata = get_convex_hull(polydata)
    bounding_box_polydata = get_bounding_box(polydata)

    # Get hull properties
    cvh_volume, cvh_surface_area = get_mass_properties(convex_hull_polydata)
    bbox_volume, bbox_surface_area = get_mass_properties(bounding_box_polydata)

    # Get centroids
    centroid = get_centroid(polydata)
    cvh_centroid = get_centroid(convex_hull_polydata)

    # Calculate features
    # All units are in units of mesh points. If mesh is generated with surface_mesh_multiscale,
    # units are physical units (um)
    volume, surface_area = get_mass_properties(polydata)
    sphericity = mesh_sphericity(volume, surface_area)
    extent = mesh_extent(volume, bbox_volume)
    solidity = mesh_solidity(volume, cvh_volume)
    concavity = mesh_concavity(volume, cvh_volume)
    asymmetry = mesh_asymmetry(volume, centroid, cvh_centroid)
    aspect_ratio = mesh_aspect_ratio(volume, convex_hull_polydata)
    surface_area_to_volume_ratio_norm = mesh_surface_area_to_volume_norm(volume, surface_area)

    # Add to feature dictionary
    vtk_measurements = {'volume': volume,
                        'surface_area': surface_area,
                        'sphericity': sphericity,
                        'extent': extent,
                        'solidity': solidity,
                        'concavity': concavity,
                        'asymmetry': asymmetry,
                        'aspect_ratio': aspect_ratio,
                        'sa_to_vol_ratio_norm': surface_area_to_volume_ratio_norm,
                        }

    return vtk_measurements, convex_hull_polydata, bounding_box_polydata
