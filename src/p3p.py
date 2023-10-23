import os
import numpy as np
from numpy.linalg import norm, inv, det


def expand_vector_dim(points):
    """
    Expands a vector from n dimensions to n+1 dimensions.
    """
    return np.hstack([points, np.ones((points.shape[0], 1))])


def unit_vec(vec):
    return vec / norm(vec)


def cos_theta(vec1, vec2):
    return np.dot(vec1, vec2) / norm(vec1) / norm(vec2)


def translate_vector(point, vector):
    return point.T - vector.reshape((3, 1))


class P3P:
    def __init__(self, camera_matrix, dist_coeffs):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        
    def trilateration(self, P, D):
        P1 = P[0]
        P2 = P[1]
        P3 = P[2]
        r1 = D[0]
        r2 = D[1]
        r3 = D[2]

        p1 = np.array([0, 0, 0])
        p2 = np.array([P2[0] - P1[0], P2[1] - P1[1], P2[2] - P1[2]])
        p3 = np.array([P3[0] - P1[0], P3[1] - P1[1], P3[2] - P1[2]])

        v1 = p2 - p1
        v2 = p3 - p1

        Xn = unit_vec(v1)
        Zn = unit_vec(np.cross(v1, v2))
        Yn = np.cross(Xn, Zn)

        i = np.dot(Xn, v2)
        d = np.dot(Xn, v1)
        j = np.dot(Yn, v2)

        X = ((r1 ** 2) - (r2 ** 2) + (d ** 2)) / (2 * d)
        Y = (((r1 ** 2) - (r3 ** 2) + (i ** 2) + (j ** 2)) / (2 * j)) - ((i / j) * (X))
        Z1 = np.sqrt(max(0, r1 ** 2 - X ** 2 - Y ** 2))
        Z2 = -Z1
        
        K1 = P1 + X * Xn + Y * Yn + Z1 * Zn
        K2 = P1 + X * Xn + Y * Yn + Z2 * Zn

        return K1, K2

    def solve_length(self, points3D, v):
        x1, x2, x3 = points3D

        Rab = norm(x1 - x2)
        Rac = norm(x1 - x3)
        Rbc = norm(x2 - x3)

        Cab = cos_theta(v[0], v[1])
        Cac = cos_theta(v[0], v[2])
        Cbc = cos_theta(v[1], v[2])

        if np.abs(Rac) < 1e-6 or np.abs(Rab) < 1e-6:
            return None, None, None

        K1 = (Rbc / Rac) ** 2
        K2 = (Rbc / Rab) ** 2
        K1K2 = K1 * K2

        G4 = (K1K2 - K1 - K2)**2 - 4 * K1K2 * Cbc**2
        G3 = 4 * (K1K2 - K1 - K2) * K2 * (1 - K1) * Cab + 4 * K1 * Cbc * ((K1K2- K1 + K2) * Cac + 2 * K2 * Cab * Cbc)
        G2 = (2 * K2 * (1 - K1) * Cab)**2 + 2 * (K1K2 - K1 - K2) * (K1K2 + K1 - K2) + 4 * K1 * ((K1 - K2) * Cbc ** 2 + K1 * (1 - K2) * Cac**2 - 2 * (1 + K1) * K2 * Cab * Cac * Cbc)
        G1 = 4 * (K1K2 + K1 - K2) * K2 * (1 - K1) * Cab + 4 * K1 * ((K1K2 - K1 + K2) * Cac * Cbc + 2 * K1K2 * Cab * Cac**2)
        G0 = (K1K2 + K1 - K2)**2 - 4 * K1**2 * K2 * Cac**2
        G = [G4, G3, G2, G1, G0]

        x = np.roots(G)
        x = x[np.isreal(x)].real

        a_square = Rab ** 2 / (1 + x ** 2 - 2 * x * Cab)
        a = np.sqrt(a_square)
        a = a[np.isreal(a)].real

        m = 1 - K1
        p = 2 * (K1 * Cac - x * Cbc)
        q = x ** 2 - K1
        m_prime = 1
        p_prime = 2 * (- x * Cbc)
        q_prime = x ** 2 * (1 - K2) + 2 * x * K2 * Cab - K2
        y = (m * q_prime - m_prime * q) / (p * m_prime - p_prime * m)        

        b = x * a
        c = y * a

        return a, b, c

    def solve(self, points3D, points2D, check_V, checkX):
        check_V = expand_vector_dim(check_V)
        v = expand_vector_dim(points2D)

        a, b, c = self.solve_length(points3D, v)
        if a is None:
            return 0, 0

        best_R, best_t = None, None
        min_dist = np.inf

        for i in range(len(a)):
            translation = self.trilateration(points3D, [a[i], b[i], c[i]])
            translation = np.array(translation)

            lambda_signs = [[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1], [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]]
            for translation_idx in range(translation.shape[0]):
                lambdas = lambda_signs * norm(translate_vector(points3D, translation[translation_idx]), axis=0) / norm(v.T, axis=0)

                for lambda_i in lambdas:
                    x_T = translate_vector(points3D, translation[translation_idx])

                    try:
                        R = (lambda_i * v.T) @ inv(x_T)
                    except np.linalg.LinAlgError:
                        continue

                    det_R_abs = np.abs(det(R))
                    if np.abs(det_R_abs - 1) < 1e-3 and np.allclose(np.dot(R.T, R), [[1, 0, 0], [0, 1, 0], [0, 0, 1]]):
                        check_x_T = translate_vector(checkX, translation[translation_idx])
                        v_pred = np.dot(R, check_x_T)

                        for lambda_element in lambda_i:
                            dist = norm(v_pred - lambda_element * check_V.T)

                            if dist < min_dist:
                                min_dist = dist
                                best_R = R
                                best_t = np.array(translation[translation_idx]).reshape(3, 1)

        return best_R, best_t
