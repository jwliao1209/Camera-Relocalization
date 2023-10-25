import numpy as np


class CameraModel:
    def __init__(self, camera_matrix, rotation_matrix, translation_vector):
        self.camera_matrix = camera_matrix
        self.rotation_matrix = rotation_matrix
        self.translation_vector = translation_vector
        self.M = camera_matrix @ np.concatenate([rotation_matrix, translation_vector], axis=1)

    def transform(self, points):
        points4D = expand_vector_dim(points) if points.shape[1] == 3 else points
        return reduce_vector_dim((self.M @ points4D.T).T)


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


def translate_vector(point, vector):
    return point - vector.reshape((1, 3))


def unitize_vector(vec):
    return vec / np.linalg.norm(vec)


def cos_theta(vec1, vec2):
    return np.dot(vec1, vec2) / np.linalg.norm(vec1) / np.linalg.norm(vec2)


def convert_intrinsic_matrix_to_4d(R, t):
    M = np.concatenate([R, t], axis=1)
    return np.concatenate([M, np.array([0, 0, 0, 1]).reshape(1, 4)], axis=0)
