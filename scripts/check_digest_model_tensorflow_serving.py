import json
import re
import time
from pathlib import Path
from typing import List

import numpy
import pandas
import requests
from PIL import Image
from matplotlib import pyplot
from object_detection.utils import visualization_utils
from tqdm import tqdm

WEAPON_CLASSES = [104, 195, 339, 369, 392, 405, 459]

NUDITY_CLASSES = [32, 203, 284]

FACE_CLASSES = [1, 3, 4, 7, 12, 21, 39, 55, 62, 64, 113]

CAR_CLASSES = [8, 14, 16, 26, 88, 87, 93, 102, 107, 214, 251, 436, 446]

APPROVED_CLASSES = [32, 203, 284, 8, 14, 16, 26, 88, 87, 93, 102, 107, 214, 251, 436, 446, 104, 195, 339, 369, 392, 405,
                    459, 1, 3, 4, 7, 12, 21, 39, 55, 62, 64, 113]

CATEGORY_DICT = {1: "car", 2: "drug", 3: "face", 4: "nudity", 5: "upskirt", 6: "weapon"}

CATEGORY_INDEX = {1: {'id': 1, 'name': 'car'}, 2: {'id': 2, 'name': 'drug'}, 3: {'id': 3, 'name': 'face'},
                  4: {'id': 4, 'name': 'nudity'}, 5: {'id': 5, 'name': 'upskirt'}, 6: {'id': 6, 'name': 'weapon'}}

DETECTION_MAP_REGEX_PATTERN = re.compile(
    "{\\'filename\\': (\\'[\w\d\-._$%@=?\\\\:]*\\'), \\'folder\\': (\\'[\w\d\-._]*\\'), \\'full path\\': (\'[\w\d\-._$%@=?\\\\:]*\'), \\'detection class\\': (\d), \\'detection class name\\': (\\'[\w]*\\'), \\'detection box\\': array\(\[(([\w.]*\s*,\s*[\w.]*\s*,\s*[\w.]*\s*,\s*[\w.]*\s*)|)\], dtype=float32\), \\'detection score\\': ([\d.]*)}")

API_ENDPOINT = "http://localhost:8501/v1/models/digest_model:predict"


def load_image_into_numpy_array(path_to_image: Path) -> numpy.ndarray:
    image = Image.open(str(path_to_image))
    (im_width, im_height) = image.size
    array_image = numpy.array(image.getdata())
    if (len(array_image.shape) < 2) or (array_image.shape[1] > 3):
        image = image.convert("RGB")
        array_image = numpy.array(image.getdata())
    return array_image.reshape((im_height, im_width, 3)).astype(numpy.uint8)


def covert_class(detected_class: int) -> int:
    return 1 if detected_class in CAR_CLASSES else (
        3 if detected_class in FACE_CLASSES else (
            4 if detected_class in NUDITY_CLASSES else (
                6 if detected_class in WEAPON_CLASSES else 0)))


def handle_detection(label_dict: dict, detection_map: list, file: Path, folder_name: str) -> List[dict]:
    label_object_list = list()

    for i in range(label_dict['num_detections']):
        detected_class = label_dict["detection_classes"][i]
        if detected_class in APPROVED_CLASSES:
            converted_class = covert_class(detected_class)
            labeled_object = {"filename": file.name, "folder": folder_name,
                              "full path": str(file.absolute()),
                              "detection class": converted_class,
                              "detection class name": CATEGORY_DICT[converted_class],
                              "detection box": label_dict["detection_boxes"][i],
                              "detection score": label_dict["detection_scores"][i]}
            detection_map.append(labeled_object)
            label_object_list.append(labeled_object)
            with Path("label_images").joinpath("detection_temp.txt").open(mode="a") as p:
                p.write(str(labeled_object))
                p.write("\n")

    if not label_object_list:
        labeled_object = {"filename": file.name, "folder": folder_name,
                          "full path": str(file.absolute()),
                          "detection class": 0,
                          "detection class name": "",
                          "detection box": [],
                          "detection score": 0.0}
        detection_map.append(labeled_object)
        with Path("label_images").joinpath("detection_temp.txt").open(mode="a") as p:
            p.write(str(labeled_object))
            p.write("\n")
    return label_object_list


if __name__ == '__main__':
    Path("label_images").mkdir(exist_ok=True)
    detection_map = list()
    detection_map_file = Path("label_images").joinpath("detection_temp.txt")
    times = Path("label_images").joinpath("times.txt")

    already_detected_files = set()
    if detection_map_file.exists():
        with detection_map_file.open(mode="r") as f:
            lines = f.readlines()
        for line in lines:
            m = DETECTION_MAP_REGEX_PATTERN.match(line)
            if m:
                detection_map.append(
                    {"filename": eval(m.group(1)), "folder": eval(m.group(2)), "full path": eval(m.group(3)),
                     "detection class": eval(m.group(4)), "detection class name": eval(m.group(5)),
                     "detection box": numpy.fromstring(m.group(6), dtype=numpy.float32, sep=','),
                     "detection score": eval(m.group(8))})
                already_detected_files.add(eval(m.group(1)))

    for folder in Path("__file__").parent.absolute().parent.joinpath("images").iterdir():
        print(folder.name)
        Path("label_images").joinpath(folder.name).mkdir(exist_ok=True)
        for file in tqdm(folder.iterdir(), unit="image", total=len(list(folder.glob('*')))):
            if file.is_file() and not (file.name in already_detected_files):
                image_np = load_image_into_numpy_array(file)
                image_np_expanded = numpy.expand_dims(image_np, axis=0)
                data = {"instances": image_np_expanded.tolist()}
                start = time.time()
                response = json.loads(requests.post(url=API_ENDPOINT, json=data).text)
                end = time.time()
                output_dict = {}
                output_dict['num_detections'] = int(response["predictions"][0]["num_detections"])
                output_dict['detection_classes'] = response["predictions"][0]["detection_classes"]
                output_dict['detection_boxes'] = response["predictions"][0]["detection_boxes"]
                output_dict['detection_scores'] = response["predictions"][0]["detection_scores"]
                with times.open(mode="a") as t:
                    t.write(f"{end - start}")
                    t.write("\n")
                label_object_list = handle_detection(output_dict, detection_map, file, folder.name)
                visualization_utils.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    numpy.asarray([x["detection box"] for x in label_object_list]),
                    numpy.asarray([x["detection class"] for x in label_object_list]),
                    numpy.asarray([x["detection score"] for x in label_object_list]),
                    CATEGORY_INDEX,
                    use_normalized_coordinates=True,
                    line_thickness=8)
                pyplot.imshow(image_np)
                file_suffix = file.suffix
                if file_suffix == ".gif":
                    file_suffix = ".jpg"
                pyplot.savefig(Path("label_images").joinpath(folder.name).joinpath(file.stem + file_suffix))

    data_frame = pandas.DataFrame(detection_map)
    data_frame.to_csv("label_images\\detection_frame.csv", index=False)
