import cv2
import numpy as np

from src.dlt import DLTRANSAC


def pnpsolver(query, model):
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

    points2D = np.empty((0,2))
    points3D = np.empty((0,3))

    for mat in gmatches:
        query_idx = mat.queryIdx
        model_idx = mat.trainIdx
        points2D = np.vstack((points2D,kp_query[query_idx]))
        points3D = np.vstack((points3D,kp_model[model_idx]))

    cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])
    distCoeffs = np.array([0.0847023,-0.192929,-0.000201144,-0.000725352])
    # return cv2.solvePnPRansac(points3D, points2D, cameraMatrix, distCoeffs)
    
    dlt_ransac = DLTRANSAC(cameraMatrix, distCoeffs)
    R, t, _ = dlt_ransac.solve(points3D, points2D)
    return R, t
