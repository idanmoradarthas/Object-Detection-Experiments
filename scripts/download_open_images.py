import multiprocessing
import multiprocessing.dummy as mp
from pathlib import Path
from typing import List
from urllib.request import urlretrieve

import pandas
from tqdm import tqdm

DOWNLOAD_URL = "https://requestor-proxy.figure-eight.com/figure_eight_datasets/open-images/"


def process(set_name: str, required_labels: List[str]) -> None:
    print(f"Working on set {set_name}")
    annotations_bbox = pandas.read_csv(f"E:\\Open Images Dataset V5\\{set_name}-annotations-bbox.csv")
    annotations_bbox_only_cars = annotations_bbox[annotations_bbox["LabelName"].isin(required_labels)].copy(deep=True)
    annotations_bbox_only_cars.to_csv(f"E:\\Open Images Dataset V5\\{set_name}-annotations-bbox-only-cars.csv",
                                      index=False)
    car_images_id = annotations_bbox_only_cars["ImageID"].unique().tolist()
    print("Starting to download...")
    Path(f"E:\\Open Images Dataset V5\\{set_name}").mkdir(exist_ok=True)
    pool = mp.Pool(processes=multiprocessing.cpu_count())
    pbar = tqdm(total=len(car_images_id), unit="image")

    def update(*a):
        pbar.update()

    [pool.apply_async(urlretrieve, args=(
        f"{DOWNLOAD_URL}{set_name}/{image_id}.jpg", f"E:\\Open Images Dataset V5\\{set_name}\\{image_id}.jpg"),
                      callback=update) for image_id in car_images_id]
    pool.close()
    pool.join()


if __name__ == '__main__':
    class_descriptions_boxable = pandas.read_csv("E:\\Open Images Dataset V5\\class-descriptions-boxable.csv",
                                                 names=["label", "description"])
    cars_labels = ["Vehicle", "Car", "Bus", "Motorcycle", "Van", "Taxi", "Tank", "Ambulance", "Limousine"]
    required_labels = class_descriptions_boxable[class_descriptions_boxable["description"].isin(cars_labels)][
        "label"].to_list()
    process("train", required_labels)
    process("validation", required_labels)
    process("test", required_labels)
