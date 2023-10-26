import numpy as np
from src.transform import CameraModel
from src.error import compute_points_error


class RANSAC:
    def __init__(
        self,
        camera_matrix,
        dist_coeffs,
        ransac_times,
        chosen_points_num,
        error_thres,
        inlier_ratio_thres,
    ):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.ransac_times = ransac_times
        self.chosen_points_num = chosen_points_num
        self.error_thres = error_thres
        self.inlier_ratio_thres = inlier_ratio_thres

    def solve(self, points3D, points2D):
        best_inliers_num = 0
        best_error = np.Inf

        for i in range(self.ransac_times):
            chosen_idx = np.random.choice(np.arange(points3D.shape[0]), size=(self.chosen_points_num, ), replace=False)
            R, t = self.solver.solve(points3D[chosen_idx], points2D[chosen_idx])

            if R is None:
                continue

            camera_model = CameraModel(self.camera_matrix, R, t)
            points2D_target = camera_model.transform(points3D)
            errors = compute_points_error(points2D_target, points2D, mean=False)
            inlier_index = np.where(errors < self.error_thres)[0]
            inlier_num = inlier_index.shape[0]
            inlier_ratio = inlier_num / points3D.shape[0]
            errors_mean = np.mean(errors)

            if i == 0 or (errors_mean < best_error and inlier_ratio > self.inlier_ratio_thres):
                best_error = errors_mean
                best_inliers_num = inlier_num
                best_R = R
                best_t = t

        return best_R, best_t, best_inliers_num
