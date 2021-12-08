
import os
import json
import tqdm

from shutil import copyfile


MAIN_PATH = "/Users/matyasbohacek/Documents/Academics/Projects/WLASL/start_kit"
BATCH = "train"

if not os.path.exists(MAIN_PATH + "/" + BATCH + "_preprocessed/"):
    os.mkdir(MAIN_PATH + "/" + BATCH + "_preprocessed/")

with open(MAIN_PATH + "/specs.json") as f:
  data = json.load(f)

for item_index, item in tqdm.tqdm(enumerate(data)):

    for video in item["instances"]:

        if video["split"] != BATCH:
            continue

        if not os.path.exists(MAIN_PATH + "/" + BATCH + "_preprocessed/" + str(item_index) + "/"):
            os.mkdir(MAIN_PATH + "/" + BATCH + "_preprocessed/" + str(item_index) + "/")

        original_path = MAIN_PATH + "/videos/" + str(video["video_id"]) + ".mp4"
        new_path = MAIN_PATH + "/" + BATCH + "_preprocessed/" + str(item_index) + "/" + str(video["video_id"]) + ".mp4"

        copyfile(original_path, new_path)

