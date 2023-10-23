import os
import cv2
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm

from src.pnp import pnpsolver
from src.cube import get_cube_points
from src.transform import CameraModel
from src.constants import CAMERA_MATRIX
from src.utils import set_random_seeds, average_desc, get_valid_id, images_to_video


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="dlt",
                        help="method of PnP")
    return parser.parse_args()


if __name__ == "__main__":
    set_random_seeds()
    args = parse_arguments()
    images_df = pd.read_pickle(os.path.join("data", "images.pkl"))
    train_df = pd.read_pickle(os.path.join("data", "train.pkl"))
    points3D_df = pd.read_pickle(os.path.join("data", "points3D.pkl"))
    point_desc_df = pd.read_pickle(os.path.join("data", "point_desc.pkl"))

    desc_df = average_desc(train_df, points3D_df)
    kp_model = np.array(desc_df["XYZ"].to_list())
    desc_model = np.array(desc_df["DESCRIPTORS"].to_list()).astype(np.float32)
    valid_ids = get_valid_id(images_df)

    image_list = []
    R_list = []
    t_list = []

    for idx in tqdm(valid_ids):
        # Load quaery image
        fname = ((images_df.loc[images_df["IMAGE_ID"] == idx])["NAME"].values)[0]
        rimg = cv2.imread("data/frames/" + fname)

        # Load query keypoints and descriptors
        points = point_desc_df.loc[point_desc_df["IMAGE_ID"] == idx]
        kp_query = np.array(points["XY"].to_list())
        desc_query = np.array(points["DESCRIPTORS"].to_list()).astype(np.float32)

        # Find correspondance and solve pnp
        R, t = pnpsolver((kp_query, desc_query), (kp_model, desc_model), method=args.method)

        image_list.append(rimg)
        R_list.append(R)
        t_list.append(t)

    point_list = get_cube_points(11)
    save_image_list = []
    for image, R, t in zip(image_list, R_list, t_list):
        camera_model = CameraModel(CAMERA_MATRIX, R, t)
        for point in point_list:
            point2D = np.int_(camera_model.transform(point.coordinate))[0]
            image = cv2.circle(image, center=point2D, radius=5, color=point.color, thickness=-1)
        save_image_list.append(image)
        
    os.makedirs("demo", exist_ok=True)
    images_to_video(save_image_list, save_path="demo/video.mp4")
