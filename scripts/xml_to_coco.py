import xmltodict
import json
import glob
import os
import sys
import numpy as np
from skimage import io
import random

def polygon_area(x,y):
	return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def get_coco_json_format():
	return {
		'info': {},
		'licenses': [],
		'images': [],
		'categories': [{}],
		'annotations': []
	}

def create_annotation_format(contour, bbox, area, image_id_, category_id, annotation_id):
    return {
        'iscrowd': 0,
        'id': annotation_id,
        'image_id': image_id_,
        'category_id': category_id,
        'bbox': bbox,
        'area': area,
        'segmentation': [contour],
    }

def create_image_annotation(image_id, file_name, width, height):
    return {
        'id': image_id,
        'width': width,
        'height': height,
        'file_name': file_name,
    }

def create_category_annotation(category_dict):
    category_list = []
    for key, value in category_dict.items():
        category = {
			'id': value,
			'name': key,
			'supercategory': key
		}
        category_list.append(category)
    return category_list

def custom_to_coco_format(xml_filenames: list[str], imgs_path: str, classes: dict[str, int]):
	annot_dict = {}
	coco_format = get_coco_json_format()
	annotation_id : int = 0
	image_id: int = 0

	coco_format["categories"] = create_category_annotation(classes)

	for xml_filename in xml_filenames:
		no_exts = xml_filename.split('/')[-1].replace('.xml', '')
		xml_annotation = None

		with open(xml_filename, 'r') as reader:
			xml_annotation = xmltodict.parse(reader.read())

		width = None
		height = None

		try:
			height, width = io.imread(os.path.join(imgs_path, xml_annotation['annotation']['filename'])).shape[:2]
		except:
			print(f"Não foi possível obter o tamanho da imagem {os.path.join(imgs_path, xml_annotation['annotation']['filename'])}")
			continue

		coco_format['images'].append(create_image_annotation(image_id, xml_annotation['annotation']['filename'], width, height))
		
		objs = xml_annotation["annotation"]["object"]

		for idz, obj in enumerate(objs):
			if obj["deleted"] != "0":
				continue # pula este objeto caso ele esteja marcado como excluido

			class_ = obj["name"]
			if class_[:5] == "tooth":
				class_ = "tooth"

			if not(class_ in classes.keys()):
				print(f"{class_} não reconhecida!")
				continue

			polygon = obj["polygon"]
			annos = polygon["pt"]

			px = []
			py = []

			try:
				for position, anno in enumerate(annos):
					px.append(int(anno["x"]))
					py.append(int(anno["y"]))
			except:
				print(f"Não foi possível obter as coordenadas do {obj['name']} para a imagem {xml_filename}")
				continue

			area = polygon_area(px, py)
			bbox = [int(np.min(px)), int(np.min(py)), int(np.max(px)), int(np.max(py))]
			poly = [(float(x), float(y)) for x, y in zip(px, py)] 
			poly = [p for x in poly for p in x]

			obj_coco = create_annotation_format(poly, bbox, area, image_id, classes[class_], annotation_id)
			annotation_id += 1
			coco_format["annotations"].append(obj_coco)

		image_id += 1

	return coco_format

if __name__ == "__main__":
	classes = {
		"tooth": 0,
		"restoration": 1,
		"root canal treatment": 2,
		"dental implant": 3,
		"crown": 4,
		"pulp": 5
	}

	xml_filenames = glob.glob('/mnt/sda1/gabriellb/Documentos/Faculdade/projetos/gaia/DeepRAD/DeepRAD/Annotations/annotations/*')
	imgs_path = "/mnt/sda1/gabriellb/Documentos/Faculdade/projetos/gaia/DeepRAD/DeepRAD/Images/images"
	output_path = "/mnt/sda1/gabriellb/Documentos/Faculdade/projetos/gaia/DeepRAD/DeepRAD/Dataset/coco"
	random.shuffle(xml_filenames)
	ds_split = (0.8, 0.2)

	coco_format_train = custom_to_coco_format(xml_filenames[:int(ds_split[0] * len(xml_filenames))], imgs_path, classes)
	coco_format_val = custom_to_coco_format(xml_filenames[int(ds_split[0] * len(xml_filenames)):], imgs_path, classes)

	coco_format_train["info"].update({
		"version": "1.0",
		"description": "X-Ray bitewing images from DeepRAD project at Universidade Federal de Pelotas (UFPel) - Brazil",
		"url": "https://github.com/glbessa/DeepRAD"
	})

	coco_format_val["info"].update({
		"version": "1.0",
		"description": "X-Ray bitewing images from DeepRAD project at Universidade Federal de Pelotas (UFPel) - Brazil",
		"url": "https://github.com/glbessa/DeepRAD"
	})

	with open(os.path.join(output_path, 'deeprad_train.json'), "w") as writer:
		writer.write(json.dumps(coco_format_train))

	with open(os.path.join(output_path, 'deeprad_val.json'), "w") as writer:
		writer.write(json.dumps(coco_format_val))
