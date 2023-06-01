# Copyright (C) 2023 Friedrich Miescher Institute for Biomedical Research

##############################################################################
#                                                                            #
# Author: Nicole Repina              <nicole.repina@fmi.ch>                  #
# Author: Manan Lalit                <lalit@mpi-cbg.de>                      #
# Author: Enrico Tagliavini          <enrico.tagliavini@fmi.ch>              #
#                                                                            #
##############################################################################

import time
import warnings

import numpy as np
import os.path
import SimpleITK as sitk
import sys
import tifffile
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

# this is an hack to run platymatch without modifying its code. In some parts of the code
# platymatch will import other submodules from itself in an absolute way (i.e. from platymatch.xxx import ....)
# rather than a relative way. Maybe we can create a pull request to switch those import statements to relative
import scmultiplex
sys.path.append(os.path.join(scmultiplex.__path__[0], r'platymatch'))

from platymatch.estimate_transform.apply_transform import (
    apply_affine_transform,
)
from platymatch.estimate_transform.perform_icp import perform_icp
from platymatch.estimate_transform.shape_context import (
    do_ransac_complete,
    get_unary,
)
from platymatch.utils.utils import (
    compute_average_bg_intensity,
    generate_affine_transformed_image,
    generate_ffd_transformed_image,
    get_centroid,
    get_mean_distance,
    normalize_min_max_percentile,
)

warnings.filterwarnings("ignore")


def calculate_stats(pc, column=-1):
    mean = np.mean(pc[:, column].transpose())
    std = np.std(pc[:, column].transpose())
    return mean, std


def calculate_size(mean, std):
    if (mean - 3 * std) > 0:
        size = mean - 3 * std
    else:
        size = mean - 2 * std
    return size


def calculate_nucleus_size(moving_pc, fixed_pc):
    moving_nuclei_size_mean, moving_nuclei_size_std = calculate_stats(moving_pc, column=-1)
    moving_nuclei_size = calculate_size(moving_nuclei_size_mean, moving_nuclei_size_std)

    fixed_nuclei_size_mean, fixed_nuclei_size_std = calculate_stats(fixed_pc, column=-1)
    fixed_nuclei_size = calculate_size(fixed_nuclei_size_mean, fixed_nuclei_size_std)

    return moving_nuclei_size, fixed_nuclei_size


def runPM(
    moving_pc,
    fixed_pc,
    ransac_iterations,
    icp_iterations,
    moving_raw_image,
    fixed_raw_image,
    moving_label_image,
    fixed_label_image,
    save_image_name,
    save_images=False,
):
    start_time = time.time()

    #     moving_pc_file_name = os.environ.get('moving_pc_file_name')
    #     fixed_pc_file_name = os.environ.get('fixed_pc_file_name')
    #     gt_matches_file_name = os.environ.get('gt_matches_file_name')
    #     moving_raw_image_name = os.environ.get('moving_raw_image_name')
    #     fixed_raw_image_name = os.environ.get('fixed_raw_image_name')
    #     moving_label_image_name = os.environ.get('moving_label_image_name')
    #     fixed_label_image_name = os.environ.get('fixed_label_image_name')

    #     save_images = False if os.environ.get('save_images') == 'False' else True
    #     if os.environ.get('save_image_name') is not None:
    #         save_affine_raw_image_name = os.environ.get('save_image_name')[:-4] + '_affine_raw.tif'
    #         save_ffd_raw_image_name = os.environ.get('save_image_name')[:-4] + '_affine_ffd_raw.tif'
    #         save_affine_label_image_name = os.environ.get('save_image_name')[:-4] + '_affine_label.tif'
    #         save_ffd_label_image_name = os.environ.get('save_image_name')[:-4] + '_affine_ffd_label.tif'

    if save_images:
        save_affine_raw_image_name = save_image_name + "_affine_raw.tif"
        save_ffd_raw_image_name = save_image_name + "_affine_ffd_raw.tif"
        save_affine_label_image_name = save_image_name + "_affine_label.tif"
        save_ffd_label_image_name = save_image_name + "_affine_ffd_label.tif"

    #     ransac_iterations = int(os.environ.get('ransac_iterations'))
    #     icp_iterations = int(os.environ.get('icp_iterations'))

    #     # Load point clouds
    #     moving_pc = pd.read_csv(moving_pc_file_name, sep=' ',
    #                             header=None).to_numpy()  # N x 5 (first column is ids, last column is size )
    #     fixed_pc = pd.read_csv(fixed_pc_file_name, sep=' ', header=None).to_numpy()
    #     matching_moving_fixed_ids = pd.read_excel(gt_matches_file_name).to_numpy()

    # Obtain ids
    moving_ids = moving_pc[:, 0].transpose()  # 1 x N
    moving_detections = np.flip(moving_pc[:, 1:-1], 1).transpose()  # z y x, 3 x N

    fixed_ids = fixed_pc[:, 0].transpose()
    fixed_detections = np.flip(fixed_pc[:, 1:-1], 1).transpose()

    # Calculate nucleus size
    moving_nuclei_size, fixed_nuclei_size = calculate_nucleus_size(moving_pc, fixed_pc)

    ransac_error = 0.5 * (moving_nuclei_size ** (1 / 3) + fixed_nuclei_size ** (1 / 3))
    print("=" * 25)
    print(f"Calculated approximate nucleus radius equal to {ransac_error:.3f}")

    # Determine centroids
    moving_centroid = get_centroid(moving_detections, transposed=False)
    fixed_centroid = get_centroid(fixed_detections, transposed=False)

    # Get Average Mean Distance
    moving_mean_distance = get_mean_distance(moving_detections, transposed=False)
    fixed_mean_distance = get_mean_distance(fixed_detections, transposed=False)

    # Generate Unaries
    print("=" * 25)
    print("Generating Unaries")

    unary_11, unary_12, _, _ = get_unary(
        moving_centroid,
        mean_distance=moving_mean_distance,
        detections=moving_detections,
        type="moving",
        transposed=False,
    )
    unary_21, unary_22, unary_23, unary_24 = get_unary(
        fixed_centroid,
        mean_distance=fixed_mean_distance,
        detections=fixed_detections,
        type="fixed",
        transposed=False,
    )

    # Generate distance matrices
    U11 = -np.matmul(unary_11, unary_21.transpose())
    U12 = -np.matmul(unary_11, unary_22.transpose())
    U13 = -np.matmul(unary_11, unary_23.transpose())
    U14 = -np.matmul(unary_11, unary_24.transpose())
    U21 = -np.matmul(unary_12, unary_21.transpose())
    U22 = -np.matmul(unary_12, unary_22.transpose())
    U23 = -np.matmul(unary_12, unary_23.transpose())
    U24 = -np.matmul(unary_12, unary_24.transpose())

    # Perform RANSAC

    transform_matrix_shape_context, inliers = do_ransac_complete(
        U11,
        U12,
        U13,
        U14,
        U21,
        U22,
        U23,
        U24,
        moving_detections,
        fixed_detections,
        ransac_samples=4,
        ransac_iterations=ransac_iterations,
        ransac_error=ransac_error,
    )
    print(
        "RANSAC # Inliers 11 = {} and # Inliers 12 = {} and # Inliers 13 = {} and # Inliers 14 = {} "
        "and # Inliers 21 = {} and # Inliers 22 = {} and # Inliers 23 = {} and # Inliers 24 = {}".format(
            *inliers
        )
    )

    sign_id = np.argmax(inliers)

    transformed_moving_detections = apply_affine_transform(
        moving_detections, transform_matrix_shape_context
    )
    transform_matrix_icp, icp_residuals = perform_icp(
        transformed_moving_detections, fixed_detections, icp_iterations, "Affine"
    )
    transform_matrix_combined = np.matmul(
        transform_matrix_icp, transform_matrix_shape_context
    )

    # Apply Affine Transform
    transformed_moving_detections = apply_affine_transform(
        moving_detections, transform_matrix_combined
    )  # 3 x N
    cost_matrix = cdist(
        transformed_moving_detections.transpose(), fixed_detections.transpose()
    )

    # `row_ids` and `col_ids` are the actual ids
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # Save to csv file, the predicted matches ids and confidence
    if sign_id == 0:
        confidence = -U11
    elif sign_id == 1:
        confidence = -U12
    elif sign_id == 2:
        confidence = -U13
    elif sign_id == 3:
        confidence = -U14
    elif sign_id == 4:
        confidence = -U21
    elif sign_id == 5:
        confidence = -U22
    elif sign_id == 6:
        confidence = -U23
    elif sign_id == 7:
        confidence = -U24

    results_affine = np.column_stack(
        (
            moving_ids[row_indices].transpose(),
            fixed_ids[col_indices].transpose(),
            confidence[row_indices, col_indices],
        )
    )
    #     np.savetxt(os.path.dirname(save_affine_label_image_name) + '/results_affine.csv', results_affine, fmt='%i %i %1.4f',
    #                header='moving_id fixed_id confidence', comments='')

    # Generate Transformed Affine Image
    (
        transformed_affine_raw_image,
        transformed_affine_label_image,
    ) = generate_affine_transformed_image(
        transform_matrix=transform_matrix_combined,
        fixed_raw_image=fixed_raw_image,
        moving_raw_image=moving_raw_image,
        moving_label_image=moving_label_image,
    )

    if save_images:
        tifffile.imsave(save_affine_raw_image_name, transformed_affine_raw_image)
        print("=" * 25)
        print(
            "Saving Affinely Transformed Raw Image with shape {} at location: {} ".format(
                transformed_affine_raw_image.shape, save_affine_raw_image_name
            )
        )

        tifffile.imsave(save_affine_label_image_name, transformed_affine_label_image)
        print(
            "Saving Affinely Transformed Label Image with shape {} at location: {} ".format(
                transformed_affine_label_image.shape, save_affine_label_image_name
            )
        )

    print("=" * 25)
    print(
        "Total Time to Run Code for Affine step is {:.3f} seconds".format(
            time.time() - start_time
        )
    )

    # Generate Free Form Deformation Image (based on Intensity Correlation) --> Note this may take some time (~5 min) [OPTIONAL]
    start_time = time.time()
    #     print("=" * 25)
    #     print("Estimating Free Form Deformation Transform")

    fixed_raw_image_normalized = normalize_min_max_percentile(
        fixed_raw_image, 1, 99.8, axis=(0, 1, 2)
    )
    transformed_affine_raw_image_normalized = normalize_min_max_percentile(
        transformed_affine_raw_image, 1, 99.8, axis=(0, 1, 2)
    )

    transform = generate_ffd_transformed_image(
        fixed_image=sitk.GetImageFromArray(
            fixed_raw_image_normalized.astype(np.float32)
        ),
        moving_image=sitk.GetImageFromArray(
            transformed_affine_raw_image_normalized.astype(np.float32)
        ),
    )

    transformed_ffd_raw_image_sitk = sitk.Resample(
        sitk.GetImageFromArray(transformed_affine_raw_image),
        sitk.GetImageFromArray(fixed_raw_image),
        transform,
        sitk.sitkLinear,
        compute_average_bg_intensity(fixed_raw_image, fixed_label_image),
        sitk.GetImageFromArray(transformed_affine_raw_image).GetPixelID(),
    )
    transformed_ffd_raw_image = sitk.GetArrayFromImage(
        transformed_ffd_raw_image_sitk
    ).astype(fixed_raw_image.dtype)

    transformed_ffd_label_image_sitk = sitk.Resample(
        sitk.GetImageFromArray(transformed_affine_label_image),
        sitk.GetImageFromArray(fixed_raw_image),
        transform,
        sitk.sitkNearestNeighbor,
        0.0,
        sitk.GetImageFromArray(transformed_affine_label_image).GetPixelID(),
    )
    transformed_ffd_label_image = sitk.GetArrayFromImage(
        transformed_ffd_label_image_sitk
    ).astype(fixed_label_image.dtype)

    if save_images:
        print("=" * 25)
        print(
            "Saving FFD Transformed Raw Image with shape {} at location: {} ".format(
                transformed_ffd_raw_image.shape, save_ffd_raw_image_name
            )
        )
        tifffile.imsave(save_ffd_raw_image_name, transformed_ffd_raw_image)

        print("=" * 25)
        print(
            "Saving FFD Transformed Label Image with shape {} at location: {} ".format(
                transformed_ffd_label_image.shape, save_ffd_label_image_name
            )
        )
        tifffile.imsave(save_ffd_label_image_name, transformed_ffd_label_image)

    # obtain accuracy after performing FFD
    # ids = np.unique(transformed_ffd_label_image)
    # ids = ids[ids != 0]
    ids = moving_ids
    moving_ffd_ids = []
    transformed_moving_ffd_detections = []
    for id in ids:
        z, y, x = np.where(transformed_ffd_label_image == id)
        zm, ym, xm = np.mean(z), np.mean(y), np.mean(x)
        if len(z) == 0:
            zm = (
                ym
            ) = (
                xm
            ) = 0.0  # TODO - in future, save `confidence` as a dictionary of id tuples)
        transformed_moving_ffd_detections.append(np.array([zm, ym, xm]))  # N x 3
        moving_ffd_ids.append(id)
    cost_matrix = cdist(transformed_moving_ffd_detections, fixed_detections.transpose())

    # `row_ids` and `col_ids` are the actual ids
    moving_ffd_ids = np.array(moving_ffd_ids)
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    #     # Evaluate Matching Accuracy after performing FFD
    #     hits = 0
    #     for row in matching_moving_fixed_ids:
    #         if col_ids[np.where(row_ids == row[1])] == row[0]:
    #             hits += 1
    #     print("=" * 25)
    #     print("Matching Accuracy after performing FFD non-rigid registration is {:.4f}".format(
    #         hits / len(matching_moving_fixed_ids)))
    #     print("=" * 25)
    print(
        "Total Additional Time to Run Code for FFD step is {:.3f} seconds".format(
            time.time() - start_time
        )
    )

    # Save matching ids to text file

    results_ffd = np.column_stack(
        (
            moving_ffd_ids[row_indices].transpose(),
            fixed_ids[col_indices].transpose(),
            confidence[row_indices, col_indices],
        )
    )
    #     np.savetxt(os.path.dirname(save_ffd_label_image_name) + '/results_ffd.csv', results_ffd, fmt='%i %i %1.4f',
    #                header='moving_id fixed_id confidence', comments='')

    return results_affine, results_ffd


def runAffine(moving_pc, fixed_pc, ransac_iterations, icp_iterations):
    """
    Run affine linking of two point clouds
    :moving_pc: numpy array of label centroids and volume of moving point cloud, e.g. RX
        rows are unique label ID and columns are in order:
            "nuc_id", "x_centroid", "y_centroid", "z_centroid", "volume"
        all units are in pixels
        scaling must match label image, i.e. z_centroid and volume do not take into account pixel anisotropy
    :fixed_pc: numpy array of label centroids and volume of fixed point cloud, e.g. R0
        same specifications as for moving_pc
    :ransac_iterations: integer number of RANSAC iterations
    :icp_iterations: integer number of ICP iterations
    :return:
        results_affine: numpy array of results, with column order: "fixed_label_id", "moving_label_id", "confidence"
        transform_matrix_combined: affine transformation matrix, for use in FFD linking or applying image transformation
        confidence: confidence measurement of matches for use in FFD linking
    """
    # Obtain ids
    moving_ids = moving_pc[:, 0].transpose()  # 1 x N
    moving_detections = np.flip(moving_pc[:, 1:-1], 1).transpose()  # z y x, 3 x N

    fixed_ids = fixed_pc[:, 0].transpose()
    fixed_detections = np.flip(fixed_pc[:, 1:-1], 1).transpose()

    # Calculate nucleus size
    moving_nuclei_size, fixed_nuclei_size = calculate_nucleus_size(moving_pc, fixed_pc)
    ransac_error = 0.5 * (moving_nuclei_size ** (1 / 3) + fixed_nuclei_size ** (1 / 3))

    # Determine centroids
    moving_centroid = get_centroid(moving_detections, transposed=False)
    fixed_centroid = get_centroid(fixed_detections, transposed=False)

    # Get Average Mean Distance
    moving_mean_distance = get_mean_distance(moving_detections, transposed=False)
    fixed_mean_distance = get_mean_distance(fixed_detections, transposed=False)

    # Generate Unaries
    unary_11, unary_12, _, _ = get_unary(
        moving_centroid,
        mean_distance=moving_mean_distance,
        detections=moving_detections,
        type="moving",
        transposed=False,
    )
    unary_21, unary_22, unary_23, unary_24 = get_unary(
        fixed_centroid,
        mean_distance=fixed_mean_distance,
        detections=fixed_detections,
        type="fixed",
        transposed=False,
    )

    # Generate distance matrices
    U11 = -np.matmul(unary_11, unary_21.transpose())
    U12 = -np.matmul(unary_11, unary_22.transpose())
    U13 = -np.matmul(unary_11, unary_23.transpose())
    U14 = -np.matmul(unary_11, unary_24.transpose())
    U21 = -np.matmul(unary_12, unary_21.transpose())
    U22 = -np.matmul(unary_12, unary_22.transpose())
    U23 = -np.matmul(unary_12, unary_23.transpose())
    U24 = -np.matmul(unary_12, unary_24.transpose())

    # Perform RANSAC
    transform_matrix_shape_context, inliers = do_ransac_complete(
        U11,
        U12,
        U13,
        U14,
        U21,
        U22,
        U23,
        U24,
        moving_detections,
        fixed_detections,
        ransac_samples=4,
        ransac_iterations=ransac_iterations,
        ransac_error=ransac_error,
    )

    sign_id = np.argmax(inliers)

    transformed_moving_detections = apply_affine_transform(
        moving_detections, transform_matrix_shape_context
    )
    transform_matrix_icp, icp_residuals = perform_icp(
        transformed_moving_detections, fixed_detections, icp_iterations, "Affine"
    )
    transform_matrix_combined = np.matmul(
        transform_matrix_icp, transform_matrix_shape_context
    )

    # Apply Affine Transform
    transformed_moving_detections = apply_affine_transform(
        moving_detections, transform_matrix_combined
    )  # 3 x N
    cost_matrix = cdist(
        transformed_moving_detections.transpose(), fixed_detections.transpose()
    )

    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # Save to csv file, the predicted matches ids and confidence
    if sign_id == 0:
        confidence = -U11
    elif sign_id == 1:
        confidence = -U12
    elif sign_id == 2:
        confidence = -U13
    elif sign_id == 3:
        confidence = -U14
    elif sign_id == 4:
        confidence = -U21
    elif sign_id == 5:
        confidence = -U22
    elif sign_id == 6:
        confidence = -U23
    elif sign_id == 7:
        confidence = -U24

    results_affine = np.column_stack(
        (
            moving_ids[row_indices].transpose(),
            fixed_ids[col_indices].transpose(),
            confidence[row_indices, col_indices],
        )
    )

    return results_affine, transform_matrix_combined, confidence


def runFFD(
    moving_pc,
    fixed_pc,
    moving_raw_image,
    fixed_raw_image,
    moving_label_image,
    fixed_label_image,
    transform_matrix_combined,
    confidence,
):
    # Generate Free Form Deformation Image (based on Intensity Correlation) --> Note this may take some time (~5 min) [OPTIONAL]
    fixed_ids = fixed_pc[:, 0].transpose()
    fixed_detections = np.flip(fixed_pc[:, 1:-1], 1).transpose()

    # Generate Transformed Affine Image
    (
        transformed_affine_raw_image,
        transformed_affine_label_image,
    ) = generate_affine_transformed_image(
        transform_matrix=transform_matrix_combined,
        fixed_raw_image=fixed_raw_image,
        moving_raw_image=moving_raw_image,
        moving_label_image=moving_label_image,
    )

    fixed_raw_image_normalized = normalize_min_max_percentile(
        fixed_raw_image, 1, 99.8, axis=(0, 1, 2)
    )
    transformed_affine_raw_image_normalized = normalize_min_max_percentile(
        transformed_affine_raw_image, 1, 99.8, axis=(0, 1, 2)
    )

    transform = generate_ffd_transformed_image(
        fixed_image=sitk.GetImageFromArray(
            fixed_raw_image_normalized.astype(np.float32)
        ),
        moving_image=sitk.GetImageFromArray(
            transformed_affine_raw_image_normalized.astype(np.float32)
        ),
    )

    transformed_ffd_label_image_sitk = sitk.Resample(
        sitk.GetImageFromArray(transformed_affine_label_image),
        sitk.GetImageFromArray(fixed_raw_image),
        transform,
        sitk.sitkNearestNeighbor,
        0.0,
        sitk.GetImageFromArray(transformed_affine_label_image).GetPixelID(),
    )
    transformed_ffd_label_image = sitk.GetArrayFromImage(
        transformed_ffd_label_image_sitk
    ).astype(fixed_label_image.dtype)

    # obtain accuracy after performing FFD
    ids = moving_pc[:, 0].transpose() # same as moving_ids in the above runPM and runAffine functions
    moving_ffd_ids = []
    transformed_moving_ffd_detections = []

    for id in ids:
        z, y, x = np.where(transformed_ffd_label_image == id)
        zm, ym, xm = np.mean(z), np.mean(y), np.mean(x)
        if len(z) == 0:
            zm = ym = xm = 0.0 # TODO - in future, save `confidence` as a dictionary of id tuples)
        transformed_moving_ffd_detections.append(np.array([zm, ym, xm]))  # N x 3
        moving_ffd_ids.append(id)
    cost_matrix = cdist(transformed_moving_ffd_detections, fixed_detections.transpose())

    # `row_ids` and `col_ids` are the actual ids
    moving_ffd_ids = np.array(moving_ffd_ids)
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # Save matching ids to text file

    results_ffd = np.column_stack(
        (
            moving_ffd_ids[row_indices].transpose(),
            fixed_ids[col_indices].transpose(),
            confidence[row_indices, col_indices],
        )
    )

    return results_ffd
