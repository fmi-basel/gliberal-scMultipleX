import numpy as np
from platymatch.estimate_transform.apply_transform import apply_affine_transform
from platymatch.estimate_transform.find_transform import (
    get_affine_transform,
    get_similar_transform,
)
from platymatch.utils.utils import get_error
from scipy.spatial import distance_matrix


def perform_icp(moving, fixed, icp_iterations=50, transform="Affine"):
    if moving.shape[0] == 4:  # 4 x N
        moving = moving[:3, :]  # 3 x N
    if fixed.shape[0] == 4:  # 4 x N
        fixed = fixed[:3, :]  # 3 x N

    if transform == "Affine":
        get_transform = get_affine_transform
        apply_transform = apply_affine_transform
    elif transform == "Similar":
        get_transform = get_similar_transform
        apply_transform = apply_similar_transform

    A_icp = np.identity(4)
    residuals = np.zeros(icp_iterations)
    for i in range(icp_iterations):
        cost_matrix = distance_matrix(
            moving.transpose(), fixed.transpose()
        )  # (N x 3, N x 3)
        i2 = np.argmin(cost_matrix, 1)
        A_est = get_transform(moving, fixed[:, i2])
        moving = apply_transform(moving, A_est)
        residuals[i] = get_error(moving, fixed[:, i2])
        A_icp = np.matmul(A_est, A_icp)
    return A_icp, residuals
