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
    
            
def get_tps_transform(moving, fixed, reg = 0.0):
    """
    :param moving: point cloud N x 3 [x, y, z]
    :param fixed: point cloud N x 3
    """
    
    N = moving.shape[0]
    # Solving A x = B
    # Create `A`
    A = np.zeros((N+4, N+4))
    k = distance_matrix(moving, moving)
    K = get_U(k) + reg*np.identity(N)
    A[:N, :N] = K
    extra_one_col = np.ones((N, 1))
    P = np.hstack((extra_one_col, moving)) # N x 4 
    PT = P.transpose() # 4 x N
    A[:N, N:] = P # N x 4
    A[N:, :N] = PT # 4 x N 
    # Create `B`
    B = np.zeros((N+4, 3))
    Bx, By, Bz = B[:, 0:1], B[:, 1:2], B[:, 2:3]
    Bx[:N, :] = fixed[:, 0:1] # N x 1 
    By[:N, :] = fixed[:, 1:2] # N x 1
    Bz[:N, :] = fixed[:, 2:3] # N x 1
    # find solution
    w_a_x= np.matmul(np.linalg.inv(A), Bx) # N x 1
    w_a_y= np.matmul(np.linalg.inv(A), By) # N x 1
    w_a_z= np.matmul(np.linalg.inv(A), Bz) # N x 1
    return w_a_x, w_a_y, w_a_z

def get_affine_transform(moving, fixed, with_ones=False):
    """
    :param moving: point cloud 3 x N
    :param fixed: point cloud 3 x N
    :param with_ones: False
    :return: 4 x 4 affine matrix
    """
    if (with_ones):
        pass
    else:
        extra_one_row = np.ones((1, moving.shape[1]))
        moving = np.vstack((moving, extra_one_row))
        fixed = np.vstack((fixed, extra_one_row))
    return np.matmul(fixed, np.linalg.pinv(moving))


# http://www.sci.utah.edu/~shireen/pdfs/tutorials/Elhabian_ICP09.pdf
def get_similar_transform(moving, fixed):
    """
    :param moving: point cloud 3 x N
    :param fixed: point cloud 3 x N
    :return: s (1) , R (3 x 3) , t (3 x 1)
    """
    com_target = np.mean(fixed, 1, keepdims=True)  # 3 x 1
    com_source = np.mean(moving, 1, keepdims=True)  # 3 x 1

    # eliminate translation
    Yprime = fixed[:3, :] - com_target[:3, :]
    Pprime = moving[:3, :] - com_source[:3, :]

    # use quarternions
    Px = Pprime[0, :]
    Py = Pprime[1, :]
    Pz = Pprime[2, :]

    Yx = Yprime[0, :]
    Yy = Yprime[1, :]
    Yz = Yprime[2, :]

    Sxx = np.sum(Yx * Px)
    Sxy = np.sum(Px * Yy)
    Sxz = np.sum(Px * Yz)

    Syx = np.sum(Py * Yx)
    Syy = np.sum(Py * Yy)
    Syz = np.sum(Py * Yz)

    Szx = np.sum(Pz * Yx)
    Szy = np.sum(Pz * Yy)
    Szz = np.sum(Pz * Yz)

    Nmatrix = [[Sxx + Syy + Szz, Syz - Szy, -Sxz + Szx, Sxy - Syx],
               [-Szy + Syz, Sxx - Szz - Syy, Sxy + Syx, Sxz + Szx],
               [Szx - Sxz, Syx + Sxy, Syy - Szz - Sxx, Syz + Szy],
               [-Syx + Sxy, Szx + Sxz, Szy + Syz, Szz - Syy - Sxx]]

    [V, D] = np.linalg.eig(Nmatrix)  # TODO values, vectors
    # optimal quarternion vector corresponds to the largest eigen value
    idx = V.argsort()[::-1]
    V = V[idx]
    D = D[:, idx]

    q = D[0]  # first value is largest value!
    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]

    Qbar = [[q0, -q1, -q2, -q3],
            [q1, q0, q3, -q2],
            [q2, -q3, q0, q1],
            [q3, q2, -q1, q0]]

    Q = [[q0, -q1, -q2, -q3],
         [q1, q0, -q3, q2],
         [q2, q3, q0, -q1],
         [q3, -q2, q1, q0]]

    R = np.matmul(np.transpose(Qbar), Q)
    R = R[1:, 1:]

    # compute scaling
    Sp = 0
    D = 0

    for i in range(Yprime.shape[1]):
        D += np.matmul(np.transpose(Yprime[:, i]), Yprime[:, i])
        Sp += np.matmul(np.transpose(Pprime[:, i]), Pprime[:, i])

    s = np.sqrt(D / Sp)
    t = com_target[:3, :] - s * np.matmul(R, com_source[:3, :])
    A = np.zeros((4, 4))
    A[:3, :3] = s * R
    A[:3, 3:4] = t
    A[3, 3] = 1
    return A
