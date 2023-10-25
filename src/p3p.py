import cv2
import numpy as np
from src.error import compute_points_error
from src.transform import expand_vector_dim, unitize_vector, cos_theta, translate_vector


class P3P:
    def __init__(self, camera_matrix, dist_coeffs):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

    def trilaterate3D(self, points, distances):
        p1, p2, p3 = points
        r1, r2, r3 = distances

        v1 = unitize_vector(p2 - p1)
        v2 = unitize_vector(p3 - p1)

        ix = v1
        iz = unitize_vector(np.cross(v1, v2))
        iy = np.cross(ix, iz)

        x2 = np.linalg.norm(p2 - p1)
        x3 = (p3 - p1) @ ix
        y3 = (p3 - p1) @ iy

        x_len = (r1**2 - r2**2 + x2**2) / (2 * x2)
        y_len = (r1**2 - r3**2 + x3**2 + y3**2 - 2 * x3 * x_len) / (2 * y3)
        z_len = np.sqrt(np.clip(r1**2 - x_len**2 - y_len**2, a_min=0, a_max=np.Inf))

        direction_vec1 = x_len * ix + y_len * iy + z_len * iz
        direction_vec2 = x_len * ix + y_len * iy - z_len * iz

        return p1 + direction_vec1, p1 + direction_vec2

    def solve_polynomial_real_roots(self, Gs):
        x = np.roots(Gs)
        return x[np.isreal(x)].real

    def check_orthogonal(self, M):
        return np.isclose(np.abs(np.linalg.det(M)), 1, atol=1e-3)

    def solve(self, points3D, points2D):
        points2D = expand_vector_dim(points2D)
        points2D, v_check = points2D[:3], points2D[-1]
        points3D, x_check = points3D[:3], points3D[-1]

        # 1. Compute G0, G1, G2, G3, G4 from the intrinsic camera matrix K and correspondences points3D and points2D
        Rab = np.linalg.norm(points3D[0] - points3D[1])
        Rac = np.linalg.norm(points3D[0] - points3D[2])
        Rbc = np.linalg.norm(points3D[1] - points3D[2])

        if np.abs(Rac) < 1e-6 or np.abs(Rab) < 1e-6:
            return None

        Cab = cos_theta(points2D[0], points2D[1])
        Cac = cos_theta(points2D[0], points2D[2])
        Cbc = cos_theta(points2D[1], points2D[2])

        K1 = (Rbc / Rac) ** 2
        K2 = (Rbc / Rab) ** 2
        K1K2 = K1 * K2

        G4 = (K1K2 - K1 - K2)**2 - 4 * K1K2 * Cbc**2
        G3 = 4 * (K1K2 - K1 - K2) * K2 * (1 - K1) * Cab + 4 * K1 * Cbc * ((K1K2- K1 + K2) * Cac + 2 * K2 * Cab * Cbc)
        G2 = (2 * K2 * (1 - K1) * Cab)**2 + 2 * (K1K2 - K1 - K2) * (K1K2 + K1 - K2) + 4 * K1 * ((K1 - K2) * Cbc ** 2 + K1 * (1 - K2) * Cac**2 - 2 * (1 + K1) * K2 * Cab * Cac * Cbc)
        G1 = 4 * (K1K2 + K1 - K2) * K2 * (1 - K1) * Cab + 4 * K1 * ((K1K2 - K1 + K2) * Cac * Cbc + 2 * K1K2 * Cab * Cac**2)
        G0 = (K1K2 + K1 - K2)**2 - 4 * K1**2 * K2 * Cac**2

        # 2. Find the root of G4x^4 + G3x^3 + G2x^2 + G1x + G0 = 0
        x = self.solve_polynomial_real_roots([G4, G3, G2, G1, G0])

        # 3. Compute a, b, c
        aa = Rab ** 2 / (1 + x ** 2 - 2 * x * Cab)
        a = np.sqrt(aa)
        a = a[np.isreal(a)].real

        if a is None:
            return None, None

        m = 1 - K1
        p = 2 * (K1 * Cac - x * Cbc)
        q = x ** 2 - K1
        m_prime = 1
        p_prime = 2 * (-x * Cbc)
        q_prime = x ** 2 * (1 - K2) + 2 * x * K2 * Cab - K2
        y = (m * q_prime - m_prime * q) / (p * m_prime - p_prime * m)        
        b = x * a
        c = y * a

        # 4. Compute the point T from a, b, c
        # (a) Find the plane ab, ac, bc
        # (b) Find the point T as the intersection of the three planes
        Ts = []
        for ai, bi, ci in zip(a, b, c):
            Ts.extend(np.array(self.trilaterate3D(points3D, (ai, bi, ci))))

        best_R, best_T = None, None
        best_error = np.inf

        for T in Ts:
            for sign in [1, -1]:
                # 5. Compute lambda by |lambda_i| = |x_i - T| / |v_i|
                X_minus_T = translate_vector(points3D, T)
                lambdas = sign * np.linalg.norm(X_minus_T, axis=1) / np.linalg.norm(points2D, axis=1)

                # 6. Compute R by lambda * v_i = R (x_i - T)
                if np.linalg.det(X_minus_T) < 1e-5:
                    continue
                else:
                    R = (lambdas * points2D.T) @ np.linalg.inv(X_minus_T.T)

                # Check R is orthogonal
                if not self.check_orthogonal(R):
                    continue

                # Check (x, v) satisfied P3P equation 
                v_pred = (R @ translate_vector(x_check, T).T).T

                for lambda_i in lambdas:
                    error = compute_points_error(v_pred, lambda_i * v_check)

                    if error < best_error:
                        best_error = error
                        best_R = R
                        best_T = T.reshape(3, 1)

        return best_R, best_T


class P3PRANSAC:
    def __init__(
        self,
        camera_matrix,
        dist_coeffs,
        ransac_times=20,
        error_thres=0.1,
    ):
        self.solver = P3P(camera_matrix, dist_coeffs)
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.ransac_times = ransac_times
        self.error_thres = error_thres
        self.chosen_points_num = 4

    def solve(self, points3D, points2D):
        '''
        Suppose the system is lambda_i * u_i = K @ R @ [I -T] X_i, i = 1, 2, 3
        (It's equivalence to lambda_i * u_i = K @ [R -RT] X_i)
        '''
        best_inliers_num = 0
        points2D = cv2.undistortPoints(points2D, self.camera_matrix, self.dist_coeffs).reshape(points2D.shape[0], 2)

        for _ in range(self.ransac_times):
            chosen_idx = np.random.choice(np.arange(points3D.shape[0]), size=(self.chosen_points_num, ), replace=False)
            R, T = self.solver.solve(points3D[chosen_idx], points2D[chosen_idx])

            if R is None:
                continue

            V = expand_vector_dim(points2D)
            X_minus_T = translate_vector(points3D, T)
            lambda_times_V = (R @ X_minus_T.T).T

            scale_factors = lambda_times_V / V
            scale_factor_max = scale_factors.max(axis=1)
            scale_factor_min = scale_factors.min(axis=1)
            scale_factor_error = (scale_factor_max - scale_factor_min) / scale_factor_max
            
            inlier_index = np.where(scale_factor_error < self.error_thres)[0]
            inlier_num = inlier_index.shape[0]

            if inlier_num > best_inliers_num:
                best_inliers_num = inlier_num
                best_R = R
                best_T = T

        return best_R, -best_R @ best_T, best_inliers_num
