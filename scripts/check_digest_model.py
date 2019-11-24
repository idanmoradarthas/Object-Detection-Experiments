import re
from pathlib import Path
from typing import List

import numpy
import pandas
import tensorflow
from PIL import Image
from matplotlib import pyplot
from object_detection.utils import visualization_utils

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


def load_graph(path_to_frozen_graph: Path) -> tensorflow.Graph:
    detection_graph = tensorflow.Graph()
    with detection_graph.as_default():
        od_graph_def = tensorflow.compat.v1.GraphDef()
        with tensorflow.io.gfile.GFile(str(path_to_frozen_graph.absolute()), "rb") as g_file:
            serialized_graph = g_file.read()
            od_graph_def.ParseFromString(serialized_graph)
            tensorflow.import_graph_def(od_graph_def, name="")
    return detection_graph


def load_image_into_numpy_array(path_to_image: Path) -> numpy.ndarray:
    image = Image.open(str(path_to_image))
    (im_width, im_height) = image.size
    array_image = numpy.array(image.getdata())
    if (len(array_image.shape) < 2) or (array_image.shape[1] > 3):
        image = image.convert("RGB")
        array_image = numpy.array(image.getdata())
    return array_image.reshape((im_height, im_width, 3)).astype(numpy.uint8)


def label_image(detection_graph: tensorflow.Graph, tensor_dict: dict, image_np: numpy.ndarray) -> dict:
    with tensorflow.compat.v1.Session(graph=detection_graph) as sess:
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        image_np_expanded = numpy.expand_dims(image_np, axis=0)
        output_dict = sess.run(tensor_dict, feed_dict={image_tensor: image_np_expanded})
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(numpy.int64)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    return output_dict


def create_tensor_dict(detection_graph: tensorflow.Graph) -> dict:
    tensor_dict = {}
    with tensorflow.compat.v1.Session(graph=detection_graph):
        ops = detection_graph.get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes']:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = detection_graph.get_tensor_by_name(tensor_name)
    return tensor_dict


def covert_class(detected_class: int) -> int:
    return 1 if detected_class in CAR_CLASSES else (
        3 if detected_class in FACE_CLASSES else (
            4 if detected_class in NUDITY_CLASSES else (
                6 if detected_class in WEAPON_CLASSES else 0)))


def handle_detection(label_dict: dict, detection_map: list, file: Path,
                     folder_name: str) -> List[dict]:
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
                          "detection box": numpy.array([], dtype=numpy.float32),
                          "detection score": 0.0}
        detection_map.append(labeled_object)
        with Path("label_images").joinpath("detection_temp.txt").open(mode="a") as p:
            p.write(str(labeled_object))
            p.write("\n")
    return label_object_list


if __name__ == '__main__':
    path_to_labels = Path("__file__").parent.absolute().parent.joinpath("models").joinpath("research").joinpath(
        "object_detection").joinpath("data").joinpath("oid_bbox_trainable_label_map.pbtxt")

    model_name = "faster_rcnn_inception_resnet_v2_atrous_lowproposals_oid_2018_01_28"
    path_to_frozen_graph = Path("__file__").parent.absolute().parent.joinpath("downloaded_models").joinpath(
        model_name).joinpath("frozen_inference_graph.pb")

    detection_graph = load_graph(path_to_frozen_graph)
    tensor_dict = create_tensor_dict(detection_graph)
    print("finished loading graph")

    Path("label_images").mkdir(exist_ok=True)
    detection_map = list()
    detection_map_file = Path("label_images").joinpath("detection_temp.txt")
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
        for file in folder.iterdir():
            if file.is_file() and not (file.name in already_detected_files):
                image_np = load_image_into_numpy_array(file)
                label_dict = label_image(detection_graph, tensor_dict, image_np)
                label_object_list = handle_detection(label_dict, detection_map, file, folder.name)
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
    # data_frame.to_feather("label_images\\detection_frame.feather")
