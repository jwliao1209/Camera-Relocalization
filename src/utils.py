import re
import numpy as np


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
