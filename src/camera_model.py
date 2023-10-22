import numpy as np
from src.dlt import expand_vector_dim, reduce_vector_dim


class CameraModel:
    def __init__(self, camera_matrix, rotation_matrix, translation_vector):
        self.camera_matrix = camera_matrix
        self.rotation_matrix = rotation_matrix
        self.translation_vector = translation_vector
        self.M = camera_matrix @ np.concatenate([rotation_matrix, translation_vector], axis=1)

    def transform(self, points):
        points4D = expand_vector_dim(points) if points.shape[1] == 3 else points
        return reduce_vector_dim((self.M @ points4D.T).T)
