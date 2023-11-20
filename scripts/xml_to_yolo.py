import xmltodict
import yaml
import glob
import os
import sys
import numpy as np
from skimage import io
import random

def get_yolo_format():
	return {
		"path":"data",
		"train":"",
		"val":"",
		"test":"",
		"names":{

		}
	}

def get_yolo_augmentations():
	return {
		"augmentation": {
			"flipud": 0.5,
			"fliplr": 0.5,
			"degrees": 90.0,
			"degrees": 180.0,
			"degrees": 270.0
		}
	}

def custom_to_yolo_format(xml_filenames: list[str], imgs_path: str, classes: dict[str, int], output_path:str, ds_type:str):
	if not os.path.exists(os.path.join(output_path, "data", ds_type)):
		os.makedirs(os.path.join(output_path, "data", ds_type))
	else:
		os.system(f"rm -r {os.path.join(output_path, 'data', ds_type, '*')}")

	for xml_filename in xml_filenames:
		no_exts = xml_filename.split('/')[-1].replace('.xml', '')
		os.system(f"cp {os.path.join(imgs_path, f'{no_exts}.jpg')} {os.path.join(output_path, 'data', ds_type)}")
		width = None
		height = None

		try:
			height, width = io.imread(os.path.join(imgs_path, f'{no_exts}.jpg')).shape[:2]
		except:
			print(f"Não foi possível obter o tamanho da imagem {os.path.join(imgs_path, f'{no_exts}.jpg')}")
			continue

		with open(os.path.join(output_path, "data", ds_type, f"{no_exts}.txt"), "w") as writer:
			xml_annotation = None
			yolo_annotations = ""

			with open(xml_filename, 'r') as reader:
				xml_annotation = xmltodict.parse(reader.read())
			
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

				contours = []

				try:
					for position, anno in enumerate(annos):
						contours.append(str(int(anno["x"])/width))
						contours.append(str(int(anno["y"])/height))
				except:
					print(f"Não foi possível obter as coordenadas do {obj['name']} para a imagem {xml_filename}")
					continue

				yolo_annotations += f"{classes[class_]} " + " ".join(contours) + "\n"

			writer.writelines(yolo_annotations)

if __name__ == "__main__":
	id2label = {
		0:"tooth",
		1:"restoration",
		2:"root canal treatment",
		3:"dental implant",
		4:"crown",
		5:"pulp"
	}

	label2id = {
		"tooth": 0,
		"restoration": 1,
		"root canal treatment": 2,
		"dental implant": 3,
		"crown": 4,
		"pulp": 5
	}

	xml_filenames = glob.glob('/mnt/sda1/gabriellb/Documentos/Faculdade/projetos/gaia/DeepRAD/DeepRAD/Annotations/annotations/*')
	imgs_path = "/mnt/sda1/gabriellb/Documentos/Faculdade/projetos/gaia/DeepRAD/DeepRAD/Images/images"
	output_path = "/mnt/sda1/gabriellb/Documentos/Faculdade/projetos/gaia/DeepRAD/DeepRAD/Dataset/yolo"
	random.shuffle(xml_filenames)
	ds_split = (0.8, 0.2)

	yolo_format = get_yolo_format()
	yolo_format["train"] = "train"
	yolo_format["val"] = "val"
	yolo_format["names"].update(id2label)

	custom_to_yolo_format(xml_filenames[0:int(ds_split[0] * len(xml_filenames))], imgs_path, label2id, output_path, "train")
	custom_to_yolo_format(xml_filenames[int(ds_split[0] * len(xml_filenames)):], imgs_path, label2id, output_path, "val")

	yolo_format.update(get_yolo_augmentations())

	with open(os.path.join(output_path, "deeprad.yaml"), "w") as writer:
		yaml.dump(yolo_format, writer)