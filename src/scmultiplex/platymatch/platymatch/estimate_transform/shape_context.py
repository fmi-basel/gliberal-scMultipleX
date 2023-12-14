from multiprocessing import Pool

from scipy.optimize import linear_sum_assignment
from skimage.measure import ransac
from skimage.transform import AffineTransform
from sklearn.decomposition import PCA
from tqdm import tqdm

from platymatch.estimate_transform.apply_transform import *
from platymatch.estimate_transform.find_transform import get_affine_transform, get_similar_transform


class AffineTransform3D(AffineTransform):
    def __init__(self):
        super(AffineTransform3D, self).__init__(dimensionality=3)


def get_Y(z, x):
    y = np.cross(z, x)
    return y / np.linalg.norm(y)


def get_shape_context(neighbors, mean_dist, r_inner=1 / 8, r_outer=2, n_rbins=5, n_thetabins=6, n_phibins=12):
    """
    :param neighbors:  N-1 x 3
    :param mean_dist: 1
    :param r_inner:
    :param r_outer:
    :param n_rbins:
    :param n_thetabins:
    :param n_phibins:
    :return:
    """
    r_ = np.linalg.norm(neighbors, axis=1)
    r = r_ / mean_dist
    theta = np.arccos(neighbors[:, 2] / r_)
    phi = np.arctan2(neighbors[:, 1], neighbors[:, 0])
    phi[phi < 0] += 2 * np.pi
    r_edges = np.logspace(np.log10(r_inner), np.log10(r_outer), n_rbins)
    index = get_bin_index(r, theta, phi, r_edges, n_rbins, n_thetabins, n_phibins)

    sc = np.zeros((n_rbins * n_thetabins * n_phibins))
    u, c = np.unique(index.astype(np.int32), return_counts=True)
    sc[u] = c
    sc = sc / np.linalg.norm(sc)

    return sc


def get_bin_index(r, theta, phi, r_edges, n_rbins, n_thetabins, n_phibins):
    theta_index = theta // (np.pi / n_thetabins)
    phi_index = phi // (2 * np.pi / n_phibins)

    r_, e_ = np.meshgrid(r, r_edges)
    binIndex = np.argmin(r_ < e_, axis=0) * n_thetabins * n_phibins + theta_index * n_phibins + phi_index

    return binIndex


def transform(detection, x_vector, y_vector, z_vector, neighbors):
    """

    :param detection: 1 x 3
    :param x_vector: 1 x 3
    :param y_vector: 1 x 3
    :param z_vector: 1 x 3
    :param neighbors: (N-1) x 3
    :return:
    """
    x_1_3d = detection + x_vector
    y_1_3d = detection + y_vector
    z_1_3d = detection + z_vector
    A = np.ones((4, 4))
    A[0, :3] = detection
    A[1, :3] = x_1_3d
    A[2, :3] = y_1_3d
    A[3, :3] = z_1_3d
    A = np.transpose(A)
    B = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 1, 1, 1]])
    T = np.matmul(B, np.linalg.inv(A))
    neighbors = np.hstack((neighbors, np.ones((neighbors.shape[0], 1))))
    points_transformed = np.matmul(T, np.transpose(neighbors))
    return np.transpose(points_transformed)[:, :3]


def get_unary_distance(sc1, sc2):
    """
    :param sc1: (360, )
    :param sc2: (360, )
    :return:
    """
    idx = sc1 != sc2
    dists = (sc1[idx] - sc2[idx]) ** 2 / (sc1[idx] + sc2[idx])
    return dists.sum() * 0.5


def do_ransac(moving_all, fixed_all, min_samples=4, trials=500, error=5, transform='Affine'):
    """

    :param moving_all: 4 x N
    :param fixed_all: 4 x N
    :param min_samples:
    :param trials:
    :param error:
    :param transform:
    :return:
    """

    if moving_all.shape[0] == 4 or fixed_all.shape[0] == 4:  # 4 x N
        moving_all = moving_all[:3, :]  # 3 x N
        fixed_all = fixed_all[:3, :]  # 3 x N

    inliers_best = 0
    A_best = np.ones((4, 4))
    for i in range(trials):
        indices = np.random.choice(fixed_all.shape[1], min_samples, replace=False)
        target_chosen = fixed_all[:, indices]  # 3 x min_samples
        source_chosen = moving_all[:, indices]  # 3 x min_samples

        if transform == 'Affine':
            transform_matrix = get_affine_transform(source_chosen, target_chosen)
        elif transform == 'Similar':
            transform_matrix = get_similar_transform(source_chosen, target_chosen)
        predicted_all = apply_affine_transform(moving_all, transform_matrix)  # 3 x N
        inliers = 0
        for index in range(fixed_all.shape[1]):
            d = np.linalg.norm(fixed_all[:, index] - predicted_all[:, index])
            if (d <= error):
                inliers += 1
        if (inliers > inliers_best):
            inliers_best = inliers
            A_best = transform_matrix
    return A_best, inliers_best



def fun(args):
    return ransac(data=args["data"],
                  min_samples=args["min_samples"],
                  max_trials=args["max_trials"],
                  residual_threshold=args["residual_threshold"],
                  model_class=args["model_class"])


def do_ransac_complete(U11, U12, U13, U14, U21, U22, U23, U24, moving_detections, fixed_detections, ransac_samples,
                       ransac_iterations, ransac_error,
                       processes=8):
    indices = list(map(linear_sum_assignment, [U11, U12, U13, U14, U21, U22, U23, U24]))

    transformations = []
    inliers = []
    for row_indices, col_indices in indices:
        (model, best_inliers) = fun(
            dict(
                data=(moving_detections[:, row_indices].T, fixed_detections[:, col_indices].T),
                min_samples=ransac_samples,
                max_trials=ransac_iterations,
                residual_threshold=ransac_error,
                model_class=AffineTransform3D,
            )
        )
        if model is None:
            print('Ransac0 failed, output[0], model')

        if best_inliers is None:
            print('Ransac0 failed, output[1], best_inliers')

        transformations.append(model)
        inliers.append(best_inliers.sum())

    return transformations[np.argmax(inliers)], inliers


def do_ransac_complete_multithread(U11, U12, U13, U14, U21, U22, U23, U24, moving_detections, fixed_detections, ransac_samples,
                       ransac_iterations, ransac_error,
                       processes=8):
    indices = list(map(linear_sum_assignment, [U11, U12, U13, U14, U21, U22, U23, U24]))

    with Pool(processes=processes) as pool:
        results = [pool.apply_async(fun, [{
            "data": (moving_detections[:, row_indices].T, fixed_detections[:, col_indices].T),
            "min_samples": ransac_samples,
            "max_trials": ransac_iterations,
            "residual_threshold": ransac_error,
            "model_class": AffineTransform3D
        }]) for row_indices, col_indices in indices]

        transformations = []
        inliers = []
        for res in results:
            output = res.get()
            transformations.append(output[0])
            inliers.append(output[1].sum())

    return transformations[np.argmax(inliers)], inliers


def get_unary(centroid, mean_distance, detections, type, n_rbins=5, n_thetabins=6, n_phibins=12, transposed=False, ):

    """
    :param centroid: N x 3 or 3 x N (transposed = False)
    :param mean_distance:
    :param detections: N x 4
    :return:
    """
    if (transposed):  # N x 3
        pass
    else:  # 3 x N
        detections = detections.transpose()
        centroid = centroid.transpose()
    if detections.shape[1] == 4:  # N x 4
        detections = detections[:, :3]  # N x 3

    sc = np.zeros((detections.shape[0], n_rbins * n_thetabins * n_phibins))
    sc2 = np.zeros((detections.shape[0], n_rbins * n_thetabins * n_phibins))
    sc3 = np.zeros((detections.shape[0], n_rbins * n_thetabins * n_phibins))
    sc4 = np.zeros((detections.shape[0], n_rbins * n_thetabins * n_phibins))
    pca = PCA(n_components=3)
    pca.fit(detections)
    V = pca.components_
    x0_vector = V[0:1, :]  # 1 x 3
    x0_vector2 = -1 * x0_vector

    for queried_index, detection in enumerate(tqdm(detections, total=len(detections))):
        neighbors = np.delete(detections, queried_index, 0)  # (N-1) x 3
        z_vector = (detection - centroid) / np.linalg.norm(detection - centroid)
        x_vector = x0_vector - z_vector * np.dot(x0_vector[0, :], z_vector[0, :])  # convert to 1D vectors
        x_vector = x_vector / np.linalg.norm(x_vector)  # vector 2 --> X
        x_vector2 = x0_vector2 - z_vector * np.dot(x0_vector2[0, :], z_vector[0, :])
        x_vector2 = x_vector2 / np.linalg.norm(x_vector2)
        y_vector = get_Y(z_vector, x_vector)
        y_vector2 = get_Y(z_vector, x_vector2)

        neighbors_transformed = transform(detection, x_vector, y_vector, z_vector, neighbors)  # (N-1) x 3
        neighbors_transformed2 = transform(detection, x_vector2, y_vector2, z_vector, neighbors)  # (N-1) x 3
        if type == 'fixed':
            y_vector3 = -y_vector
            y_vector4 = -y_vector2
            neighbors_transformed3 = transform(detection, x_vector, y_vector3, z_vector, neighbors)  # (N-1) x 3
            neighbors_transformed4 = transform(detection, x_vector2, y_vector4, z_vector, neighbors)  # (N-1) x 3
            sc3[queried_index] = get_shape_context(neighbors_transformed3, mean_distance, n_rbins=n_rbins,
                                                   n_thetabins=n_thetabins, n_phibins=n_phibins)
            sc4[queried_index] = get_shape_context(neighbors_transformed4, mean_distance, n_rbins=n_rbins,
                                                   n_thetabins=n_thetabins, n_phibins=n_phibins)
        sc[queried_index] = get_shape_context(neighbors_transformed, mean_distance, n_rbins=n_rbins,
                                              n_thetabins=n_thetabins, n_phibins=n_phibins)
        sc2[queried_index] = get_shape_context(neighbors_transformed2, mean_distance, n_rbins=n_rbins,
                                               n_thetabins=n_thetabins, n_phibins=n_phibins)
    return sc, sc2, sc3, sc4
