import yaml
import json
import os
import math
import glob
from PIL import Image
from logging import Logger

class Annotation:
    def __init__(self):
        pass

class SegmentationAnnotation(Annotation):
    def __init__(self, class_id, points: list[tuple[float]] | list[tuple[int]]):
        self.class_id = class_id
        self.points = points

class ImageAnnotations:
    def __init__(self, image_path: str, annotations: Annotation, image_height: int, image_width: int):
        self.image_path = image_path
        self.annotations = annotations
        self.image_height = image_height
        self.image_width = image_width

class YOLOv8Dataset:
    def __init__(self, id2class: dict[int, str], data_path: str):
        self.id2class = id2class
        self.data_path = data_path

        self.images_filenames: list[str] = []
        self.images_dimensions: list[tuple[int]] = []
        self.images_annotations: list[dict[int, list]] = []
        self.images_dataset_type: list[str] = []

    @staticmethod
    def from_yolov8format(yaml_path: str):
        if not os.path.exists(yaml_path) or not os.path.isfile(yaml_path):
            raise Exception()
        
        yaml_file = None
        with open(yaml_path, "r") as reader:
            yaml_file = yaml.load(reader.read())

        classes = yaml_file.get('names').values()
        print(classes)

        return Dataset(classes, yaml_file.get('path'))

    @staticmethod
    def from_cocoformat(json_path: str, data_path: str):
        coco_file = None
        with open(json_path, "r") as reader:
            coco_file = json.load(reader)

        id2class = {}
        for category in coco_file.get('categories'):
            id2class.update({
                category.get('id'): category.get('name')
            })

        return Dataset(id2class, data_path)

    def to_cocoformat(self, json_path):
        pass

    def to_yolov8format(self, yaml_path):
        pass

    def _parse_segmentation_file(self, annotations_path: str):
        with open(annotations_path, 'r') as reader:
            pass

    def _parse_segmentation_line(self, annotations: str):
        annotations = annotations.split(" ")
        class_id = int(annotations.pop(0))
        points = []
        for index in range(0, len(annotations), 2):
            points.append((float(annotations[index]), float(annotations[index + 1])))

        return class_id, points

    def crop_images(self, height: int, width: int):
        pass

    def crop_images_by_masks(self, x_margin: int = 0, y_margin: int = 0):
        pass

    def _crop_single_image_by_mask(self, image_path):
        pass
        