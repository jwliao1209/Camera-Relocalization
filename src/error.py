import numpy as np


def compute_vector_2norm(vec):
    return np.sqrt((vec ** 2).sum(axis=1))


def compute_points_error(p1, p2, mean=True):
    errors = compute_vector_2norm(p1-p2)
    return errors.mean() if mean else errors
