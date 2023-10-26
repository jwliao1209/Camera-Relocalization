import cv2
import numpy as np
from src.ransac import RANSAC


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


class DLTRANSAC(RANSAC):
    def __init__(
        self,
        camera_matrix,
        dist_coeffs,
        ransac_times=20,
        chosen_points_num=100,
        error_thres=15,
        inlier_ratio_thres=0.5,
    ):
        super().__init__(
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            ransac_times=ransac_times,
            chosen_points_num=chosen_points_num,
            error_thres=error_thres,
            inlier_ratio_thres=inlier_ratio_thres,
        )
        self.solver = DLT(camera_matrix, dist_coeffs)
