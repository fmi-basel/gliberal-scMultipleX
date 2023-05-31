import numpy as np
from scipy.spatial import distance_matrix
from numpy import inf

def get_U(k):
    """
    k = N x N matrix
    """

    log_k=np.log(k)

    log_k[log_k==-inf] = 0
    return k*log_k


def apply_tps_transform(moving_all, moving_cp, w_a_x, w_a_y, w_a_z):
    """
    moving_all N x 3 
    moving control points N1 x 3 
    w_a_x N1+4 x 1 ( first N rows are weights)
    """
    extra_one_col = np.ones((moving_all.shape[0], 1))
    m1 = np.hstack((extra_one_col, moving_all)) # N x 4
    
    N = moving_all.shape[0]
    N1 = moving_cp.shape[0] 
    
    A = np.zeros((N, N1+4), dtype=np.float) # N x (N1+4)
    k = distance_matrix(moving_all, moving_cp) # N x N1
    A[:, :N1] = get_U(k) # N x N1
    A[:, N1:] = m1 # N x 4
    transformed_moving_x = np.matmul(A, w_a_x) # N x 1
    transformed_moving_y = np.matmul(A, w_a_y) # N x 1
    transformed_moving_z = np.matmul(A, w_a_z) # N x 1
    return np.hstack((transformed_moving_x, transformed_moving_y, transformed_moving_z))  # N x 3 (x1 x2 x3 ...; y1 y2 y3 ...; z1 z2 z3) 
    
def apply_affine_transform(moving, affine_transform_matrix):
    """

    :param moving: 3 x N
    :param affine_transform_matrix: 4 x 4
    :param with_ones: if False, then source is 3 x N, else 4 x N
    :return: target point cloud 3 x N
    """
    if moving.shape[0] == 4:  # 4 x N
        moving = moving[:3, :] # 3 x N

    extra_one_row = np.ones((1, moving.shape[1]))
    moving = np.vstack((moving, extra_one_row))
    fixed_predicted = np.matmul(affine_transform_matrix, moving) # 4 x N
    return fixed_predicted[:3, :]

def apply_similar_transform(source, scale, rotation, translation, with_ones=False):
    """

    :param source:
    :param scale: scalar
    :param rotation: 3 x 3
    :param translation: 3 x 1
    :param with_ones: False
    :return: target 3 x N
    """
    if (with_ones):
        source = source[:3, :]
    else:
        pass
    return scale * np.matmul(rotation, source) + translation
