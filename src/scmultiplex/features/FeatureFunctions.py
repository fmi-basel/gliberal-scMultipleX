# Copyright (C) 2023 Friedrich Miescher Institute for Biomedical Research

##############################################################################
#                                                                            #
# Author: Nicole Repina              <nicole.repina@fmi.ch>                  #
# Author: Tim-Oliver Buchholz        <tim-oliver.buchholz@fmi.ch>            #
# Author: Enrico Tagliavini          <enrico.tagliavini@fmi.ch>              #
#                                                                            #
##############################################################################

import numpy as np
import scipy
import scipy.stats
import math

from skimage.measure import marching_cubes, mesh_surface_area, label, moments, regionprops
from skimage.morphology import binary_erosion

from scmultiplex.meshing.MeshFunctions import get_max_length

spacing = None


def set_spacing(spacing_s):
    global spacing
    spacing = spacing_s


def fixed_percentiles(region_mask, intensity):
    """
    Calculate values at percentiles 25, 50, 75, 90, 95, 99 of foreground pixels of intensity image
    :param region_mask: binary foreground background mask
    :param intensity: intensity image
    :return: intensity value list[float]
    """
    assert region_mask.dtype == bool, "Region mask must be bool type"
    return np.percentile(intensity[region_mask], q=(25, 50, 75, 90, 95, 99))


def skewness(region_mask, intensity):
    """Return skewness of pixel intensity distribution of raw image masked by segmentation
    """
    return scipy.stats.skew(intensity[region_mask])


def kurtos(region_mask, intensity):
    """Return kurtosis of pixel intensity distribution of raw image masked by segmentation
    """
    return scipy.stats.kurtosis(intensity[region_mask])


def stdv(region_mask, intensity):
    """Return standard deviation of pixel intensity distribution of raw image masked by segmentation
    """
    # ddof=1 for sample var
    return np.std(intensity[region_mask], ddof=1)


def bounding_box_ratio(prop):
    """Return the ratio between the object area and the bounding box area of the object
    Note this is the same as regionprops 'extent'
    """
    return prop.area / prop.area_bbox


def convex_hull_ratio(prop):
    """Return the ratio between the object area and the convex hull area of the object
    Note this is the same as regionprops 'solidity'
    """
    return prop.area / prop.area_convex


def convex_hull_area_resid(prop_2D):
    """Return the normalized difference in area between the convex hull and area of the object
    Normalize to the area of the convex hull, becomes fraction of object composed of concavities (divots & indentations)
    """
    return (prop_2D.convex_area - prop_2D.area) / prop_2D.convex_area


# from Ark
# https://github.com/angelolab/ark-analysis/blob/main/src/ark/segmentation/regionprops_extraction.py
def convex_hull_centroid_dif(prop_2D, spacing):
    """Return the normalized euclidian distance between the centroid of the object label
        and the centroid of the convex hull
        Normalize to the object area
    """

    if len(spacing) != 2:
        raise ValueError("spacing must be 2D for convex_hull_centroid_dif calculation")

    # Use image that has same size as bounding box (not original seg)
    object_image = prop_2D.image
    object_moments = moments(object_image)
    object_centroid = np.array([object_moments[1, 0] / object_moments[0, 0], object_moments[0, 1] / object_moments[0, 0]])

    # Convex hull image has same size as bounding box
    convex_image = prop_2D.convex_image
    convex_moments = moments(convex_image)
    convex_centroid = np.array([convex_moments[1, 0] / convex_moments[0, 0], convex_moments[0, 1] / convex_moments[0, 0]])

    # rescale to correct scaling
    object_centroid_scaled = object_centroid * spacing
    convex_centroid_scaled = convex_centroid * spacing

    # calculate 2-norm (Euclidean distance) and normalize
    centroid_dist = np.linalg.norm(object_centroid_scaled - convex_centroid_scaled) / np.sqrt(prop_2D.area)

    return centroid_dist


def circularity(prop_2D):
    """Return the circularity of object
    Perfect circle has a circularity of 1; lower circularity means deviates from perfect circle
    Circularity is defined as the ratio of sample area / circle area, where sample area is the measured object area and
     circle area is the area of the circle that has the identical perimeter as the measured labeled object.
    """
    return 4 * math.pi * (prop_2D.area / np.square(prop_2D.perimeter))


def aspect_ratio(prop):
    """Return the ratio of major axis length to equivalent diameter
    """
    return prop.major_axis_length / prop.equivalent_diameter


def minor_major_axis_ratio(prop):
    """Return the ratio of major to minor axis
    """
    if prop.major_axis_length == 0:
        return np.float64('NaN')
    else:
        return prop.minor_axis_length / prop.major_axis_length


def concavity_count(prop_2D, min_area_fraction=0.005):
    """Return the number of concavities for an object
    min_area_fraction is the concavity area divided by total object area. concavities above this cutoff are counted as
    a concavity
    """
    object_image = prop_2D.image
    convex_image = prop_2D.convex_image
    object_area = prop_2D.area

    diff_img = convex_image ^ object_image  # bitwise XOR

    if np.sum(diff_img) > 0:
        labeled_diff_img = label(diff_img, connectivity=1)
        concavity_feat = regionprops(labeled_diff_img)
        concavity_cnt = 0
        for concavity_2D in concavity_feat:
            if (concavity_2D.area / object_area) > min_area_fraction:
                concavity_cnt += 1
    else:
        concavity_cnt = 0
    return concavity_cnt


def disconnected_component(mask_2D):
    """Return boolean True if disconnected components detected in input labelmap
    """
    if len(np.unique(mask_2D)) > 2:
        raise ValueError('mask must be binary')
    disconnected = False
    labeled_mask_img = label(mask_2D, connectivity=2)
    if len(np.unique(labeled_mask_img)) > 2:
        disconnected = True
    return disconnected


def surface_area_marchingcube(mask_3D):
    """Return surface area of 3D label image
    """
    regionmask_int = mask_3D.astype(np.uint16)
    # add zero-pad on all sides, otherwise mesh is smaller than it should be
    regionmask_int = np.pad(regionmask_int, 1, 'constant')
    verts, faces, normals, values = marching_cubes(regionmask_int, spacing=spacing)
    surface_area = mesh_surface_area(verts, faces)
    return surface_area


def flag_touching(ovr_seg_img, ovr_seg_tiles):
    """Return list of integer org_id's that are touching tile border in well overview image
    Calculated based on tile image of Drogon output
    """
    tile_borders = (ovr_seg_tiles - binary_erosion(ovr_seg_tiles)).astype(
        bool
    )  # generate tile borders

    touching_labels = np.unique(
        ovr_seg_img[tile_borders]
    )  # includes the 0 background label

    # create list of labels and remove 0 background label
    touching_labels_lst = [int(x) for x in touching_labels[touching_labels > 0]]

    return touching_labels_lst


def is_touching_border_xy(labeled_obj, img_shape):
    """
    Helper function to check if an object is touching the border of the image
    in the xy plane.
    """
    if len(img_shape) == 2:
        if labeled_obj.bbox[0] == 0 or labeled_obj.bbox[1] == 0:
            return True
        elif labeled_obj.bbox[2] == img_shape[0] or labeled_obj.bbox[3] == img_shape[1]:
            return True
        else:
            return False
    elif len(img_shape) == 3:
        if labeled_obj.bbox[1] == 0 or labeled_obj.bbox[2] == 0:
            return True
        elif labeled_obj.bbox[4] == img_shape[1] or labeled_obj.bbox[5] == img_shape[2]:
            return True
        else:
            return False
    else:
        raise NotImplementedError("Only 2D and 3D images are supported in is_touching_border_xy")


def is_touching_border_z(labeled_obj, img_shape):
    """
    Helper function to check if an object is touching the border of the image
    in the 3D in the Z direction.
    """
    if len(img_shape) == 3:
        if labeled_obj.bbox[0] == 0:
            return True
        elif labeled_obj.bbox[3] == img_shape[0]:
            return True
        else:
            return False
    else:
        raise NotImplementedError("Only 3D images are supported in is_touching_border_z")


def centroid_weighted_correct(labeled_obj, spacing):
    centroid_local = labeled_obj.centroid_weighted_local
    bb_origin = np.array([x.start for x in labeled_obj.slice])
    bb_origin_scaled = bb_origin * spacing
    return bb_origin_scaled + centroid_local


def centroid_weighted_correct(labeled_obj):
    centroid_local = labeled_obj.centroid_weighted_local
    return tuple(idx + slc.start * spc
                 for idx, slc, spc in zip(centroid_local, labeled_obj.slice, labeled_obj._spacing))


########################
# MESH FUNCTIONS #######
########################


def mesh_equivalent_surface_area(volume):
    """
    Calculate the surface area of a sphere with a given volume.
    """
    if volume < 0:
        raise ValueError('Cannot calculate radius of sphere with a negative volume')

    radius = (0.75 * volume * (1/math.pi)) ** (1. / 3.)
    equivalent_sa = 4 * math.pi * (radius ** 2.)

    return equivalent_sa


def mesh_equivalent_diameter(volume):
    """
    The diameter of a sphere with the same volume as the object region
    """
    equiv_diam = 2 * ((0.75 * volume * (1/math.pi)) ** (1. / 3.))
    return equiv_diam


def mesh_sphericity(volume, surface_area):
    """
    Return object sphericity given the measured object volume and surface area. Perfect sphere has a sphericity of 1;
    higher sphericity means that object shape deviates from a sphere. Sphericity is defined as the
    ratio of measured object surface area to the surface area of a sphere with an identical volume.
    This is a 3D version of circularity.
    """
    equivalent_sa = mesh_equivalent_surface_area(volume)
    if equivalent_sa == 0:
        raise ValueError('Cannot calculate sphericity for object with volume = 0')
    return surface_area / equivalent_sa


def mesh_extent(object_volume, bbox_volume):
    """Return the ratio between the object volume and the bounding box volume
    Note this is the 3D version of bounding_box_ratio (i.e. 'extent')
    """
    return object_volume / bbox_volume


def mesh_solidity(object_volume, convex_hull_volume):
    """Return the ratio between the object volume and the convex hull volume
    Note this is the 3D version of convex_hull_ratio (i.e. 'solidity')
    Convex objects have a convex_hull_volume_ratio close 1; objects with concavities have values less than 1.
    """
    return object_volume / convex_hull_volume


def mesh_concavity(object_volume, convex_hull_volume):
    """Return the normalized difference in volume between the convex hull and the object
    Normalize to the volume of the convex hull so that number reflects the fraction of object composed of
    concavities (divots & indentations). Values closer to 0 indicate a more convex shape.
    Note this is the 3D version of convex_hull_area_resid
    """
    return (convex_hull_volume - object_volume) / convex_hull_volume


def mesh_asymmetry(object_volume, object_centroid, convex_hull_centroid):
    """Return the normalized euclidian distance between the centroid of the object point cloud
        and the centroid of the convex hull
        Normalize to the cubed root of object volume
    """

    # calculate 2-norm (Euclidean distance) and normalize
    centroid_dist = np.linalg.norm(object_centroid - convex_hull_centroid) / np.cbrt(object_volume)

    return centroid_dist


def mesh_aspect_ratio(object_volume, polydata):
    """
    Return the ratio of maximum length of mesh point cloud to the equivalent diameter.
    Typically, use the convex hull vtkPolyData object as input polydata.
    """
    equiv_diam = mesh_equivalent_diameter(object_volume)
    maxdist, _ = get_max_length(polydata)

    return maxdist / equiv_diam


def mesh_surface_area_to_volume(object_volume, object_surface_area):
    """ Return surface are to volume ratio"""
    return object_surface_area / object_volume


def mesh_surface_area_to_volume_norm(object_volume, object_surface_area):
    """ Return square root of surface are to cubed root of volume ratio.
    Normalized by factor a so that value is 1.0 for spherical objects
    This metric is invariant to object size, unlike simple surface area to volume ratio
    which scales with 3/radius for spherical objects"""
    a = (3**(1/3))*((4*np.pi)**(1/6))
    ratio = (object_surface_area**(1/2)) / (object_volume**(1/3) * a)
    return ratio
