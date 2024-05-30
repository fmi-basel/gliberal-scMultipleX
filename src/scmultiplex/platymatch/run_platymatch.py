import time

import SimpleITK as sitk
import numpy as np
import os
import pandas as pd
import tifffile
import warnings
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from platymatch.estimate_transform.apply_transform import apply_affine_transform
from platymatch.estimate_transform.perform_icp import perform_icp
from platymatch.estimate_transform.shape_context import get_unary, do_ransac_complete
from platymatch.utils.utils import get_centroid, get_mean_distance, generate_affine_transformed_image, \
    generate_ffd_transformed_image, normalize_min_max_percentile, compute_average_bg_intensity

# warnings.filterwarnings("ignore")

# Typical Starter Code

# moving_pc_file_name='../data/20220111/round_one.csv' fixed_pc_file_name='../data/20220111/round_zero.csv'
# gt_matches_file_name='../data/20220111/groundtruth.xlsx' ransac_iterations='4000'  icp_iterations='50'
# moving_raw_image_name='../data/20220111/R1_65_raw.tif' fixed_raw_image_name='../data/20220111/R0_67_raw.tif'
# moving_label_image_name='../data/20220111/R1_65_pred.tif' fixed_label_image_name='../data/20220111/R0_67_pred.tif'
# save_images='False' save_image_name='../data/20220111/transformed_image.tif' python3 run_platymatch.py

# moving_pc_file_name='../data/20220124/set1/round_one.csv' fixed_pc_file_name='../data/20220124/set1/round_zero.csv'
# gt_matches_file_name='../data/20220124/set1/groundtruth_09_10.xlsx' ransac_iterations='4000'  icp_iterations='50'
# moving_raw_image_name='../data/20220124/set1/R1_10_raw.tif' fixed_raw_image_name='../data/20220124/set1/R0_09_raw.tif'
# moving_label_image_name='../data/20220124/set1/R1_10_pred.tif' fixed_label_image_name='../data/20220124/set1/R0_09_pred.tif'
# save_images='False' save_image_name='../data/20220124/set1/transformed_image.tif' python3 run_platymatch.py


# moving_pc_file_name='../data/20220124/set2/round_one.csv' fixed_pc_file_name='../data/20220124/set2/round_zero.csv'
# gt_matches_file_name='../data/20220124/set2/groundtruth_24_25.xlsx' ransac_iterations='4000'  icp_iterations='50'
# moving_raw_image_name='../data/20220124/set2/R1_25_raw.tif' fixed_raw_image_name='../data/20220124/set2/R0_24_raw.tif'
# moving_label_image_name='../data/20220124/set2/R1_25_pred.tif' fixed_label_image_name='../data/20220124/set2/R0_24_pred.tif'
# save_images='False' save_image_name='../data/20220124/set2/transformed_image.tif' python3 run_platymatch.py


start_time = time.time()

moving_pc_file_name = os.environ.get('moving_pc_file_name')
fixed_pc_file_name = os.environ.get('fixed_pc_file_name')
gt_matches_file_name = os.environ.get('gt_matches_file_name')
moving_raw_image_name = os.environ.get('moving_raw_image_name')
fixed_raw_image_name = os.environ.get('fixed_raw_image_name')
moving_label_image_name = os.environ.get('moving_label_image_name')
fixed_label_image_name = os.environ.get('fixed_label_image_name')

save_images = False if os.environ.get('save_images') == 'False' else True
if os.environ.get('save_image_name') is not None:
    save_affine_raw_image_name = os.environ.get('save_image_name')[:-4] + '_affine_raw.tif'
    save_ffd_raw_image_name = os.environ.get('save_image_name')[:-4] + '_affine_ffd_raw.tif'
    save_affine_label_image_name = os.environ.get('save_image_name')[:-4] + '_affine_label.tif'
    save_ffd_label_image_name = os.environ.get('save_image_name')[:-4] + '_affine_ffd_label.tif'

ransac_iterations = int(os.environ.get('ransac_iterations'))
icp_iterations = int(os.environ.get('icp_iterations'))

# Load point clouds
moving_pc = pd.read_csv(moving_pc_file_name, sep=' ',
                        header=None).to_numpy()  # N x 5 (first column is ids, last column is size )
fixed_pc = pd.read_csv(fixed_pc_file_name, sep=' ', header=None).to_numpy()
matching_moving_fixed_ids = pd.read_excel(gt_matches_file_name).to_numpy()

# Obtain ids
moving_ids = moving_pc[:, 0].transpose()  # 1 x N
moving_detections = np.flip(moving_pc[:, 1:-1], 1).transpose()  # z y x, 3 x N

fixed_ids = fixed_pc[:, 0].transpose()
fixed_detections = np.flip(fixed_pc[:, 1:-1], 1).transpose()

# Calculate nucleus size
moving_nuclei_size_mean = np.mean(moving_pc[:, -1].transpose())
moving_nuclei_size_std = np.std(moving_pc[:, -1].transpose())
moving_nuclei_size = moving_nuclei_size_mean - 3 * moving_nuclei_size_std if (
                                                                                     moving_nuclei_size_mean - 3 * moving_nuclei_size_std) > 0 else (
        moving_nuclei_size_mean - 2 * moving_nuclei_size_std)
fixed_nuclei_size_mean = np.mean(fixed_pc[:, -1].transpose())
fixed_nuclei_size_std = np.std(fixed_pc[:, -1].transpose())
fixed_nuclei_size = fixed_nuclei_size_mean - 3 * fixed_nuclei_size_std if (
                                                                                  fixed_nuclei_size_mean - 3 * fixed_nuclei_size_std) > 0 else (
        fixed_nuclei_size_mean - 2 * fixed_nuclei_size_std)

ransac_error = 0.5 * (moving_nuclei_size ** (1 / 3) + fixed_nuclei_size ** (1 / 3))
print("=" * 25)
print("Calculated approximate nucleus radius equal to {:.3f}".format(ransac_error))

# Determine centroids
moving_centroid = get_centroid(moving_detections, transposed=False)
fixed_centroid = get_centroid(fixed_detections, transposed=False)

# Get Average Mean Distance
moving_mean_distance = get_mean_distance(moving_detections, transposed=False)
fixed_mean_distance = get_mean_distance(fixed_detections, transposed=False)

# Load Raw Images
moving_raw_image = tifffile.imread(moving_raw_image_name)
fixed_raw_image = tifffile.imread(fixed_raw_image_name)

# Load Labels
moving_label_image = tifffile.imread(moving_label_image_name)
fixed_label_image = tifffile.imread(fixed_label_image_name)

# Generate Unaries
print("=" * 25)
print("Generating Unaries")

unary_11, unary_12, _, _ = get_unary(moving_centroid, mean_distance=moving_mean_distance, detections=moving_detections,
                                     type='moving', transposed=False)
unary_21, unary_22, unary_23, unary_24 = get_unary(fixed_centroid, mean_distance=fixed_mean_distance,
                                                   detections=fixed_detections, type='fixed', transposed=False)

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

transform_matrix_shape_context, inliers = do_ransac_complete(U11, U12, U13, U14, U21, U22, U23, U24, moving_detections,
                                                             fixed_detections, ransac_samples=4,
                                                             ransac_iterations=ransac_iterations,
                                                             ransac_error=ransac_error)
print("RANSAC # Inliers 11 = {} and # Inliers 12 = {} and # Inliers 13 = {} and # Inliers 14 = {} "
      "and # Inliers 21 = {} and # Inliers 22 = {} and # Inliers 23 = {} and # Inliers 24 = {}".format(*inliers))

sign_id = np.argmax(inliers)


transformed_moving_detections = apply_affine_transform(moving_detections, transform_matrix_shape_context)
transform_matrix_icp, icp_residuals = perform_icp(transformed_moving_detections, fixed_detections, icp_iterations,
                                                  'Affine')
transform_matrix_combined = np.matmul(transform_matrix_icp, transform_matrix_shape_context)

# Apply Affine Transform
transformed_moving_detections = apply_affine_transform(moving_detections, transform_matrix_combined)  # 3 x N
cost_matrix = cdist(transformed_moving_detections.transpose(), fixed_detections.transpose())

# `row_ids` and `col_ids` are the actual ids
row_indices, col_indices = linear_sum_assignment(cost_matrix)
row_ids = moving_ids[row_indices]
col_ids = fixed_ids[col_indices]

# Evaluate Matching Accuracy
hits = 0
for row in matching_moving_fixed_ids:
    if col_ids[np.where(row_ids == row[1])] == row[0]:
        hits += 1
print("=" * 25)
print("Matching Accuracy is {:.4f}".format(hits / len(matching_moving_fixed_ids)))

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
    (moving_ids[row_indices].transpose(), fixed_ids[col_indices].transpose(), confidence[row_indices, col_indices]))
np.savetxt(os.path.dirname(save_affine_label_image_name) + '/results_affine.csv', results_affine, fmt='%i %i %1.4f',
           header='moving_id fixed_id confidence', comments='')


# Generate Transformed Affine Image
transformed_affine_raw_image, transformed_affine_label_image = generate_affine_transformed_image(
    transform_matrix=transform_matrix_combined, fixed_raw_image=fixed_raw_image,
    moving_raw_image=moving_raw_image, moving_label_image=moving_label_image)

if save_images:
    tifffile.imsave(save_affine_raw_image_name, transformed_affine_raw_image)
    print("=" * 25)
    print("Saving Affinely Transformed Raw Image with shape {} at location: {} ".format(
        transformed_affine_raw_image.shape,
        save_affine_raw_image_name))

    tifffile.imsave(save_affine_label_image_name, transformed_affine_label_image)
    print("Saving Affinely Transformed Label Image with shape {} at location: {} ".format(
        transformed_affine_label_image.shape, save_affine_label_image_name))

print("=" * 25)
print("Total Time to Run Code for Affine step is {:.3f} seconds".format(time.time() - start_time))

# Generate Free Form Deformation Image (based on Intensity Correlation) --> Note this may take some time (~5 min) [OPTIONAL]
start_time = time.time()
print("=" * 25)
print("Estimating Free Form Deformation Transform")

fixed_raw_image_normalized = normalize_min_max_percentile(fixed_raw_image, 1, 99.8, axis=(0, 1, 2))
transformed_affine_raw_image_normalized = normalize_min_max_percentile(transformed_affine_raw_image, 1, 99.8,
                                                                       axis=(0, 1, 2))

transform = generate_ffd_transformed_image(
    fixed_image=sitk.GetImageFromArray(fixed_raw_image_normalized.astype(np.float32)),
    moving_image=sitk.GetImageFromArray(
        transformed_affine_raw_image_normalized.astype(np.float32)))

transformed_ffd_raw_image_sitk = sitk.Resample(sitk.GetImageFromArray(transformed_affine_raw_image),
                                               sitk.GetImageFromArray(fixed_raw_image),
                                               transform,
                                               sitk.sitkLinear,
                                               compute_average_bg_intensity(fixed_raw_image, fixed_label_image),
                                               sitk.GetImageFromArray(transformed_affine_raw_image).GetPixelID())
transformed_ffd_raw_image = sitk.GetArrayFromImage(transformed_ffd_raw_image_sitk).astype(fixed_raw_image.dtype)

transformed_ffd_label_image_sitk = sitk.Resample(sitk.GetImageFromArray(transformed_affine_label_image),
                                                 sitk.GetImageFromArray(fixed_raw_image),
                                                 transform,
                                                 sitk.sitkNearestNeighbor,
                                                 0.0,
                                                 sitk.GetImageFromArray(transformed_affine_label_image).GetPixelID())
transformed_ffd_label_image = sitk.GetArrayFromImage(transformed_ffd_label_image_sitk).astype(fixed_label_image.dtype)

if save_images:
    print("=" * 25)
    print("Saving FFD Transformed Raw Image with shape {} at location: {} ".format(transformed_ffd_raw_image.shape,
                                                                                   save_ffd_raw_image_name))
    tifffile.imsave(save_ffd_raw_image_name, transformed_ffd_raw_image)

    print("=" * 25)
    print("Saving FFD Transformed Label Image with shape {} at location: {} ".format(transformed_ffd_label_image.shape,
                                                                                     save_ffd_label_image_name))
    tifffile.imsave(save_ffd_label_image_name, transformed_ffd_label_image)

# obtain accuracy after performing FFD
ids = moving_ids
moving_ffd_ids = []
transformed_moving_ffd_detections = []

for id in ids:
    z, y, x = np.where(transformed_ffd_label_image == id)
    zm, ym, xm = np.mean(z), np.mean(y), np.mean(x)
    if len(z) ==0:
        zm = ym = xm = 0.0 # TODO - in future, save `confidence` as a dictionary of id tuples)
    transformed_moving_ffd_detections.append(np.array([zm, ym, xm]))  # N x 3
    moving_ffd_ids.append(id)

cost_matrix = cdist(transformed_moving_ffd_detections, fixed_detections.transpose())

# `row_ids` and `col_ids` are the actual ids
moving_ffd_ids = np.array(moving_ffd_ids)
row_indices, col_indices = linear_sum_assignment(cost_matrix)
row_ids = moving_ffd_ids[row_indices]
col_ids = fixed_ids[col_indices]

# Evaluate Matching Accuracy after performing FFD
hits = 0
for row in matching_moving_fixed_ids:
    if col_ids[np.where(row_ids == row[1])] == row[0]:
        hits += 1
print("=" * 25)
print("Matching Accuracy after performing FFD non-rigid registration is {:.4f}".format(
    hits / len(matching_moving_fixed_ids)))
print("=" * 25)
print("Total Additional Time to Run Code for FFD step is {:.3f} seconds".format(time.time() - start_time))

# Save matching ids to text file


results_ffd = np.column_stack(
    (moving_ffd_ids[row_indices].transpose(), fixed_ids[col_indices].transpose(), confidence[row_indices, col_indices]))
np.savetxt(os.path.dirname(save_ffd_label_image_name) + '/results_ffd.csv', results_ffd, fmt='%i %i %1.4f',
           header='moving_id fixed_id confidence', comments='')

