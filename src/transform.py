import cv2
import numpy as np
from src.p3p import P3P
from src.error import compute_points_error


def translate_vector(point, vector):
    return point.T - vector.reshape((3, 1))


def expand_vector_dim(points):
    """
    Expands a vector from n dimensions to n+1 dimensions.
    """
    return np.hstack([points, np.ones((points.shape[0], 1))])


def reduce_vector_dim(points):
    """
    Reduces a vector from n+1 dimensions to n dimensions.
    """
    EPSILON = 1e-8
    dim = points.shape[-1]
    points = points / (np.expand_dims(points[:, dim-1], axis=1) + EPSILON)
    return points[:, :dim-1]


def convert_intrinsic_matrix_to_4d(R, t):
    M = np.concatenate([R, t], axis=1)
    return np.concatenate([M, np.array([0, 0, 0, 1]).reshape(1, 4)], axis=0)


class CameraModel:
    def __init__(self, camera_matrix, rotation_matrix, translation_vector):
        self.camera_matrix = camera_matrix
        self.rotation_matrix = rotation_matrix
        self.translation_vector = translation_vector
        self.M = camera_matrix @ np.concatenate([rotation_matrix, translation_vector], axis=1)

    def transform(self, points):
        points4D = expand_vector_dim(points) if points.shape[1] == 3 else points
        return reduce_vector_dim((self.M @ points4D.T).T)


class DLT:
    def __init__(self, camera_matrix, dist_coeffs):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

    def compute_one_point_sub_matrix(self, x, y, z, u, v):
        """
        K = [fx, 0, cx]
            [0, fy, cy]
            [0,  0,  1]
        """
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        cx_u = cx - u
        cy_v = cy - v

        return np.array(
            [
                [x*fx, y*fx, z*fx, fx,    0,    0,    0,  0, cx_u*x, cx_u*y, cx_u*z, cx_u],
                [0,    0,    0,    0,  x*fy, y*fy, z*fy, fy, cy_v*x, cy_v*y, cy_v*z, cy_v],
            ]
        )

    def solve(self, points3D, points2D):
        points2D = cv2.undistortImagePoints(points2D, self.camera_matrix, self.dist_coeffs).reshape(points2D.shape[0], 2)

        # Solve Ax = 0 using SVD
        A = np.vstack(
            [
                self.compute_one_point_sub_matrix(x, y, z, u, v)
                for (x, y, z), (u, v) in zip(points3D, points2D)
            ]
        )

        _, _, x = np.linalg.svd(A)
        X = x[-1].reshape(3, 4)
        R_bar, t_bar = X[:, :3], X[:, 3].reshape(3, 1)
        U, S, Vh = np.linalg.svd(R_bar)
        beta = 3 / np.sum(S)

        # Compute the value of beta * (r31 * x + r32 * y + r33 * z + t3)
        if beta * np.dot(np.array([*points3D[0], 1]), X[2, :]) > 0:
            R = U @ Vh
            t = beta * t_bar
        else:
            R = -U @ Vh
            t = -beta * t_bar
        return R, t


class DLTRANSAC:
    def __init__(
        self,
        camera_matrix,
        dist_coeffs,
        ransac_times=20,
        chosen_points_num=100,
        error_thres=15,
        inlier_ratio_thres=0.5,
    ):
        self.solver = DLT(camera_matrix, dist_coeffs)
        self.camera_matrix = camera_matrix
        self.distCoeffs = dist_coeffs
        self.ransac_times = ransac_times
        self.chosen_points_num = chosen_points_num
        self.error_thres = error_thres
        self.inlier_ratio_thres = inlier_ratio_thres

    def solve(self, points3D, points2D):
        best_error = np.Inf
        best_inliers_num = 0

        for i in range(self.ransac_times):
            chosen_idx = np.random.choice(np.arange(points3D.shape[0]), size=(self.chosen_points_num, ), replace=False)
            R, t = self.solver.solve(points3D[chosen_idx], points2D[chosen_idx])
            camera_model = CameraModel(self.camera_matrix, R, t)
            points2D_target = camera_model.transform(points3D)
            errors = compute_points_error(points2D_target, points2D, mean=False)
            inlier_index = np.where(errors < self.error_thres)[0]
            inlier_num = inlier_index.shape[0]
            inlier_ratio = inlier_num / points3D.shape[0]
            errors_mean = np.mean(errors)

            if i == 1 or (errors_mean < best_error and inlier_ratio > self.inlier_ratio_thres):
                best_error = errors_mean
                best_inliers_num = inlier_num
                best_R = R
                best_t = t

        return best_R, best_t, best_inliers_num


class P3PRANSAC:
    def __init__(
        self,
        camera_matrix,
        dist_coeffs,
    ):
        self.solver = P3P(camera_matrix, dist_coeffs)
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.chosen_points_num = 3
    
    def get_ransac_times(self):
        prob = 0.99
        error_rate = 0.4
        valid_num = 3
        return int(np.ceil(np.log(1 - prob) / np.log(1 - (1 - error_rate) ** (valid_num + 3))))

    def solve(self, points3D, points2D):
        from src.constants import CAMERA_MATRIX, DIST_COEFFS

        points2D = cv2.undistortPoints(points2D, CAMERA_MATRIX, DIST_COEFFS)[:, 0]

        n_ransec_time = self.get_ransac_times()
        n_inlier = 0

        for i in range(n_ransec_time):
            chosen_idx = np.random.choice(np.arange(points3D.shape[0]), size=(self.chosen_points_num, ), replace=False)
            idx_unsample = np.full(points2D.shape[0], True)
            idx_unsample[chosen_idx] = False
            check_X = np.array([points3D[idx_unsample][0]])
            check_V = np.array([points2D[idx_unsample][0]])

            R, t = self.solver.solve(points3D[chosen_idx], points2D[chosen_idx], check_V, check_X)

            if R is None:
                continue

            v_unsample = expand_vector_dim(points2D)
            X_T_unsample = translate_vector(points3D, t)
            lambda_v = R @ X_T_unsample
            dist = lambda_v / v_unsample.T
            dist = (dist.max(axis = 0) - dist.min(axis = 0)) / dist.max(axis = 0)
            epsilon = 0.1
            if dist[dist < epsilon].shape[0] > n_inlier:
                n_inlier = dist[dist < epsilon].shape[0]
                best_R = R
                best_t = t

        return best_R, -best_R @ best_t, n_inlier
