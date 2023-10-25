import cv2
import numpy as np

from src.dlt import DLTRANSAC
from src.p3p import P3PRANSAC
from src.constants import CAMERA_MATRIX, DIST_COEFFS


def pnpsolver(query, model, method):
    kp_query, desc_query = query
    kp_model, desc_model = model

    # bf = cv2.BFMatcher()
    # matches = bf.knnMatch(desc_query, desc_model, k=2)

    flann = cv2.FlannBasedMatcher()
    matches = flann.knnMatch(desc_query, desc_model, k=2)

    gmatches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            gmatches.append(m)

    points2D = np.empty((0, 2))
    points3D = np.empty((0, 3))

    for mat in gmatches:
        query_idx = mat.queryIdx
        model_idx = mat.trainIdx
        points2D = np.vstack((points2D,kp_query[query_idx]))
        points3D = np.vstack((points3D,kp_model[model_idx]))

    # return cv2.solvePnPRansac(points3D, points2D, cameraMatrix, distCoeffs)
    
    if method == "dlt":
        pnp_ransac = DLTRANSAC(CAMERA_MATRIX, DIST_COEFFS)
    
    elif method == "p3p":
        pnp_ransac = P3PRANSAC(CAMERA_MATRIX, DIST_COEFFS)

    R, t, _ = pnp_ransac.solve(points3D, points2D)

    return R, t
