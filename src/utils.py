import re
import cv2
import random
import numpy as np


def set_random_seeds(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    return


def average(x):
    return list(np.mean(x, axis=0))


def average_desc(train_df, points3D_df):
    train_df = train_df[["POINT_ID","XYZ","RGB","DESCRIPTORS"]]
    desc = train_df.groupby("POINT_ID")["DESCRIPTORS"].apply(np.vstack)
    desc = desc.apply(average)
    desc = desc.reset_index()
    desc = desc.join(points3D_df.set_index("POINT_ID"), on="POINT_ID")
    return desc


def get_valid_id(df):
    def get_id_from_name(name):
        return int(re.search(r'valid_img(\d+).jpg', name).group(1))

    valid_df = df[df["NAME"].apply(lambda x: "valid" in x)]
    valid_df["VALID_ID"] = valid_df["NAME"].apply(lambda x: get_id_from_name(x))
    valid_df = valid_df.sort_values(by="VALID_ID")
    return valid_df["IMAGE_ID"]


def images_to_video(image_list, save_path="video.mp4"):
    video_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, (1080, 1920))
    for image in image_list:
        video_writer.write(image)
    video_writer.release()
    return
