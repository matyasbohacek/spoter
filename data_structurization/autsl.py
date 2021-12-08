
import os
import tqdm

import pandas as pd
from shutil import copyfile


MAIN_PATH = "/Users/matyasbohacek/Documents/Academics/Projects/AUTSL"
BATCH = "test"

df = pd.read_csv(MAIN_PATH + "/" + BATCH + "_labels.csv", encoding="utf-8", sep=";")

if not os.path.exists(MAIN_PATH + "/" + BATCH + "_preprocessed/"):
    os.mkdir(MAIN_PATH + "/" + BATCH + "_preprocessed/")

for index_row, row in tqdm.tqdm(df.iterrows()):
    if not os.path.exists(MAIN_PATH + "/" + BATCH + "_preprocessed/" + str(row["label"]) + "/"):
        os.mkdir(MAIN_PATH + "/" + BATCH + "_preprocessed/" + str(row["label"]) + "/")

    copyfile(MAIN_PATH + "/" + BATCH + "/" + str(row["video"]) + "_color.mp4", MAIN_PATH + "/" + BATCH + "_preprocessed/" + str(row["label"]) + "/" + str(row["video"]) + "_color.mp4")

