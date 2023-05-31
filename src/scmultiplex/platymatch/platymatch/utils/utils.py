import SimpleITK as sitk
import numpy as np
import tifffile
from scipy.interpolate import griddata
from tqdm import tqdm
from platymatch.estimate_transform.apply_transform import apply_tps_transform


def get_centroid(detections, transposed=True):
    """
    :param detections: N x 3/4 (transpposed = True) or 3/4 x N (transposed = False)
    :return: 3 x 1
    """
    if (transposed):  # N x 4
        return np.mean(detections[:, :3], 0, keepdims=True)
    else:  # 4 x N
        return np.mean(detections[:3, :], 1, keepdims=True)


def get_mean_distance(detections, transposed=True):
    """
    :param detections: 3 x N
    :return:
    """

    if (transposed):  # N x 3
        pass
    else:  # 3/4 x N as intended
        detections = detections.transpose()  # N x 3/4

    if detections.shape[1] == 4:
        detections = detections[:, :3]  # N x 3
    pair_wise_distance = []
    for i in range(detections.shape[0]):
        for j in range(i + 1, detections.shape[0]):
            pair_wise_distance.append(np.linalg.norm(detections[i] - detections[j]))
    return np.average(pair_wise_distance)


def get_error(moving_landmarks, fixed_landmarks):
    """

    :param moving_landmarks: 3 x N
    :param fixed_landmarks: 3 x N
    :return:
    """
    if (moving_landmarks is not None or fixed_landmarks is not None):
        residual = moving_landmarks - fixed_landmarks
        return np.mean(np.linalg.norm(residual, axis=0))  # axis= 0 means norm is taken along axis = 0
    else:
        return None


def affine_translate(transform, dx, dy, dz):
    new_transform = sitk.AffineTransform(transform)
    new_transform.SetTranslation((dx, dy, dz))
    return new_transform


def affine_rotate(transform, rotation, dimension=3):
    parameters = np.array(transform.GetParameters())
    new_transform = sitk.AffineTransform(transform)
    matrix = np.array(transform.GetMatrix()).reshape((dimension, dimension))
    new_matrix = np.dot(rotation, matrix)
    new_transform.SetMatrix(new_matrix.ravel())
    return new_transform


def generate_affine_transformed_image(transform_matrix, fixed_raw_image, moving_raw_image,
                                      moving_label_image):
    """
    :param transform_matrix:
    :param fixed_raw_image:
    :param moving_raw_image:
    :param moving_label_image:
    :return:
    """

    affine = sitk.AffineTransform(3)
    transform_xyz = np.zeros_like(transform_matrix)  # 4 x 4
    transform_xyz[:3, :3] = np.flip(np.flip(transform_matrix[:3, :3], 0), 1)
    transform_xyz[0, 3] = transform_matrix[2, 3]
    transform_xyz[1, 3] = transform_matrix[1, 3]
    transform_xyz[2, 3] = transform_matrix[0, 3]
    transform_xyz[3, 3] = 1.0
    inv_matrix = np.linalg.inv(transform_xyz)
    affine = affine_translate(affine, inv_matrix[0, 3], inv_matrix[1, 3], inv_matrix[2, 3])
    affine = affine_rotate(affine, inv_matrix[:3, :3])
    reference_raw_image = sitk.GetImageFromArray(fixed_raw_image)
    moving_raw_image_sitk = sitk.GetImageFromArray(moving_raw_image)
    moving_label_image_sitk = sitk.GetImageFromArray(moving_label_image)

    interpolatorLinear = sitk.sitkLinear
    interpolatorNearestNeighbor = sitk.sitkNearestNeighbor
    default_value = 0.0

    transformed_affine_raw_image = sitk.GetArrayFromImage(
        sitk.Resample(moving_raw_image_sitk, reference_raw_image, affine, interpolatorLinear, default_value))

    transformed_affine_label_image = sitk.GetArrayFromImage(
        sitk.Resample(moving_label_image_sitk, reference_raw_image, affine, interpolatorNearestNeighbor, default_value))

    return transformed_affine_raw_image, transformed_affine_label_image


def generate_tps_transformed_image(fixed_image, moving_image, moving_control_points, w_a_x, w_a_y, w_a_z,
                                   save_tps_image_name, ds_grid_factor=4):
    transformed_moving_image_coordinates = []
    for z in tqdm(range(0, moving_image.shape[0])):
        for y in range(0, moving_image.shape[1], ds_grid_factor):
            moving_image_coordinates = []
            for x in range(0, moving_image.shape[2], ds_grid_factor):
                moving_image_coordinates.append(np.array([x, y, z]))
            transformed_moving_image_coordinates.append(
                apply_tps_transform(np.asarray(moving_image_coordinates), moving_control_points, w_a_x, w_a_y,
                                    w_a_z))  # N x 3 (x1 y1 z1; x2 y2 z2; ...)
    transformed_moving_image_coordinates = np.asarray(transformed_moving_image_coordinates).reshape(
        (moving_image.shape[0]) * (moving_image.shape[1] // ds_grid_factor) * (moving_image.shape[2] // ds_grid_factor),
        3)

    data = []
    for z in range(0, moving_image.shape[0]):
        for y in range(0, moving_image.shape[1], ds_grid_factor):
            for x in range(0, moving_image.shape[2], ds_grid_factor):
                data.append(moving_image[z, y, x])
    data = np.asarray(data)

    fixed_image_coordinates = []
    for z in range(fixed_image.shape[0]):
        for y in range(fixed_image.shape[1]):
            for x in range(fixed_image.shape[2]):
                fixed_image_coordinates.append(np.array([x, y, z]))
    fixed_image_coordinates = np.asarray(fixed_image_coordinates)

    print("=" * 25)
    print("Applying TPS transform")
    transformed_tps_image = griddata(transformed_moving_image_coordinates, data, fixed_image_coordinates)
    transformed_tps_image = transformed_tps_image.reshape(fixed_image.shape[0], fixed_image.shape[1],
                                                          fixed_image.shape[2])
    tifffile.imsave(save_tps_image_name, transformed_tps_image.astype(fixed_image.dtype))
    print("=" * 25)
    print("Saving TPS Transformed Image with shape {} at location: {} ".format(transformed_tps_image.shape,
                                                                               save_tps_image_name))

    return transformed_tps_image

#https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/blob/master/Python/65_Registration_FFD.ipynb
def generate_ffd_transformed_image(fixed_image, moving_image, fixed_image_mask=None):
    registration_method = sitk.ImageRegistrationMethod()
    # Determine the number of BSpline control points using the physical spacing we want for the control grid.
    grid_physical_spacing = [50.0, 50.0, 50.0]  # A control point every 50mm
    image_physical_size = [size * spacing for size, spacing in zip(fixed_image.GetSize(), fixed_image.GetSpacing())]
    mesh_size = [int(image_size / grid_spacing + 0.5) \
                 for image_size, grid_spacing in zip(image_physical_size, grid_physical_spacing)]

    initial_transform = sitk.BSplineTransformInitializer(image1=fixed_image, transformDomainMeshSize=mesh_size, order=3)
    registration_method.SetInitialTransform(initial_transform)

    registration_method.SetMetricAsMeanSquares()
    # Settings for metric sampling, usage of a mask is optional. When given a mask the sample points will be
    # generated inside that region. Also, this implicitly speeds things up as the mask is smaller than the
    # whole image.
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    if fixed_image_mask:
        registration_method.SetMetricFixedMask(fixed_image_mask)

    # Multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=100)

    return registration_method.Execute(fixed_image, moving_image)

def normalize_min_max_percentile(x, pmin=3, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32):
    """
        Percentile-based image normalization.
    """
    mi = np.percentile(x, pmin, axis=axis, keepdims=True)
    ma = np.percentile(x, pmax, axis=axis, keepdims=True)
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)


def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
    if dtype is not None:
        x = x.astype(dtype, copy=False)
        mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
        ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
        eps = dtype(eps)

    try:
        import numexpr
        x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
    except ImportError:
        x = (x - mi) / (ma - mi + eps)

    if clip:
        x = np.clip(x, 0, 1)

    return x

def compute_average_bg_intensity(raw_image, label_image):
    return int(np.mean(raw_image[label_image==0]))