import cv2
import numpy as np
from src.transform import CameraModel
from src.error import compute_points_error


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
        self.dist_coeffs = dist_coeffs
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
