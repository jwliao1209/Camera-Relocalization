import os
import cv2
import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy.spatial.transform import Rotation

from src.pnp import pnpsolver
from src.visualize import Visualize3D
from src.transform import convert_intrinsic_matrix_to_4d
from src.utils import set_random_seeds, average_desc, get_valid_id


if __name__ == "__main__":
    set_random_seeds()
    images_df = pd.read_pickle(os.path.join("data", "images.pkl"))
    train_df = pd.read_pickle(os.path.join("data", "train.pkl"))
    points3D_df = pd.read_pickle(os.path.join("data", "points3D.pkl"))
    point_desc_df = pd.read_pickle(os.path.join("data", "point_desc.pkl"))

    # Process model descriptors
    desc_df = average_desc(train_df, points3D_df)
    kp_model = np.array(desc_df["XYZ"].to_list())
    desc_model = np.array(desc_df["DESCRIPTORS"].to_list()).astype(np.float32)

    point3D_loc = np.vstack(points3D_df['XYZ'].values)
    point3D_color = np.vstack(points3D_df['RGB'].values) / 255
    visualizer = Visualize3D(point3D_loc, point3D_color)

    valid_ids = get_valid_id(images_df)
    R_error_list = []
    t_error_list = []

    for idx in tqdm(valid_ids):
        # Load quaery image
        fname = ((images_df.loc[images_df["IMAGE_ID"] == idx])["NAME"].values)[0]
        rimg = cv2.imread("data/frames/" + fname, cv2.IMREAD_GRAYSCALE)

        # Load query keypoints and descriptors
        points = point_desc_df.loc[point_desc_df["IMAGE_ID"] == idx]
        kp_query = np.array(points["XY"].to_list())
        desc_query = np.array(points["DESCRIPTORS"].to_list()).astype(np.float32)

        # Find correspondance and solve pnp
        R, t = pnpsolver((kp_query, desc_query), (kp_model, desc_model))

        # Get camera pose groudtruth 
        ground_truth = images_df.loc[images_df["IMAGE_ID"] == idx]
        rotq_gt = ground_truth[["QX","QY","QZ","QW"]].values
        R_gt = Rotation.from_quat(rotq_gt[0]).as_matrix()
        t_gt = ground_truth[["TX","TY","TZ"]].values.reshape(3, 1)

        R_error = np.linalg.norm(Rotation.from_matrix(R @ R_gt.T).as_rotvec())
        t_error = np.linalg.norm(t - t_gt)

        R_error_list.append(R_error)
        t_error_list.append(t_error)

        M = convert_intrinsic_matrix_to_4d(R, t)
        visualizer.add_pose(M)

    print(f"Rotation Error: {np.median(R_error_list)}")
    print(f"Translation Error: {np.median(t_error_list)}")

    visualizer.draw_geometry()
