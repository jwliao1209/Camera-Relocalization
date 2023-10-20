import numpy as np


def convert_intrinsic_matrix_to_4d(R, t):
    M = np.concatenate([R, t], axis=1)
    return np.concatenate([M, np.array([0, 0, 0, 1]).reshape(1, 4)], axis=0)
