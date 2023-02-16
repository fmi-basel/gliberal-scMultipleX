import numpy as np
import scipy
from scipy.ndimage import distance_transform_edt
from skimage.measure import marching_cubes, mesh_surface_area
from skimage.morphology import binary_erosion


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
    return scipy.stats.skew(intensity[region_mask])


def kurtos(region_mask, intensity):
    return scipy.stats.kurtosis(intensity[region_mask])


def stdv(region_mask, intensity):
    # ddof=1 for sample var
    return np.std(intensity[region_mask], ddof=1)


def surface_area_marchingcube(regionmask, intensity, spacing=(3, 1, 1)):
    regionmask_int = regionmask.astype(int)
    verts, faces, normals, values = marching_cubes(
        regionmask_int, level=0, spacing=spacing
    )
    return mesh_surface_area(verts, faces)


def expand_labels(label_image, distance=1):  # from scikit
    distances, nearest_label_coords = distance_transform_edt(
        label_image == 0, return_indices=True
    )
    labels_out = np.zeros_like(label_image)
    dilate_mask = distances <= distance
    masked_nearest_label_coords = [
        dimension_indices[dilate_mask] for dimension_indices in nearest_label_coords
    ]
    nearest_labels = label_image[tuple(masked_nearest_label_coords)]
    labels_out[dilate_mask] = nearest_labels
    return labels_out


def graphical_neighbor_count(nuc_seg, nuc_id, expandby_pix=21, spacing=(3, 1, 1)):
    # isolate given nucleus and make binary
    seg_id = np.zeros_like(nuc_seg)
    seg_id[nuc_seg == nuc_id] = 1
    expandby_z = round(expandby_pix / spacing[0])  # this is how many slices in z to add
    # make binary mask of given label expanded in 3d
    expanded = np.zeros_like(seg_id)
    slices_with_label = []

    for i, zslice in enumerate(seg_id):
        if zslice[zslice > 0].size > 0:  # if a z-slice contains the nuc_id label...
            slices_with_label.append(i)  # record it

        expanded[i, :, :] = expand_labels(
            zslice, expandby_pix
        )  # expand labels in each zslice in xy

    # expand top zslice in z
    top_id = np.amax(slices_with_label)
    for t in range(
        top_id + 1, top_id + expandby_z + 1, 1
    ):  # for zslices above the top slice, expand by given distance
        if (
            t <= expanded.shape[0] - 1
        ):  # as long as index does not extend beyond image boundary
            expanded[t, :, :] = expanded[
                top_id, :, :
            ]  # replace empty z-slice with that of expanded top mask

    # expand bottom zslice in z
    bottom_id = np.amin(slices_with_label)
    for b in range(
        bottom_id - 1, (bottom_id - expandby_z) - 1, -1
    ):  # for zslices below the bottom slice, expand by given distance
        if b >= 0:  # as long as index does not extend beyond image boundary
            expanded[b, :, :] = expanded[
                bottom_id, :, :
            ]  # replace empty z-slice with that of expanded bottom mask

    # mask full nuc labelmap by this 3D expanded nucleus
    seg_masked = nuc_seg * expanded

    # count number of unique labels remaining in the labelmap
    # filter out 0 background and self
    filters = (lambda x: (x != 0), lambda x: (x != nuc_id))
    neighbor_list = list(
        filter(lambda x: all(f(x) for f in filters), np.unique(seg_masked))
    )  # this is the list nuc_ids of neighbors
    unique_neighbor_count = len(
        neighbor_list
    )  # this is the number of identified neighbors
    return unique_neighbor_count, neighbor_list


def flag_touching(ovr_seg_img, ovr_seg_tiles):
    tile_borders = (ovr_seg_tiles - binary_erosion(ovr_seg_tiles)).astype(
        bool
    )  # generate tile borders

    touching_labels = np.unique(
        ovr_seg_img[tile_borders]
    )  # includes the 0 background label

    touching_labels_lst = [
        "object_" + str(x) for x in touching_labels[touching_labels > 0]
    ]  # create list of labels and remove 0 background label

    return touching_labels_lst
