import xmltodict
import json
import glob
import os
import sys

XML_PATH = ""
COCO_PATH = ""

category_ids = {
	"restoration": 1,
	"root canal treatment": 2,
	"dental implant": 3,
	"crown": 4,
	"tooth": 5,
	"pulp": 6
}

def create_category_annotation(category_dict):
    category_list = []
    for key, value in category_dict.items():
        category = {"id": value, "name": key, "supercategory": key}
        category_list.append(category)
    return category_list

def create_image_annotation(file_name, width, height):
	image_id = file_name.split("-")[-1].split(".")[0]
    return {
        "id": image_id,
        "width": width,
        "height": height,
        "file_name": file_name,
    }


def create_annotation_format(contour, image_id_, category_id, annotation_id):
    return {
        "iscrowd": 0,
        "id": annotation_id,
        "image_id": image_id_,
        "category_id": category_id,
        "bbox": cv2.boundingRect(contour),
        "area": cv2.contourArea(contour),
        "segmentation": [contour.flatten().tolist()],
    }


def get_coco_json_format():
    return {
        "info": {},
        "licenses": [],
        "images": [{}],
        "categories": [{}],
        "annotations": [{}],
    }

def rename_xmls(xml_folder, new_folder):
	xml_files = glob.glob(os.path.join(xml_folder, '*.xml'))

	for xml_file in xml_files:
		actual_id = xml_file.split("/")[-1].split("-")[1].split(".")[0]
		future_id = int(actual_id) + 120
		new_filename = f"imagem-{str(future_id).zfill(3)}"


		xml_text = None
		with open(xml_file, "r") as reader:
			xml_text = reader.read()
		
		xml_text.replace(f"imagem-{actual_id}.jpg", f"{new_filename}.jpg")

		if os.path.exists(new_folder) == False:
			os.makedirs(new_folder)

		with open(os.path.join(new_folder, f"{new_filename}.xml"), "w") as writer:
			writer.write(xml_text)

def parse_folder_to_file(xml_path, coco_path):
	files = glob.glob(os.path.join(xml_path, '*.xml'))
	coco_format = get_coco_json_format()


	for file_ in files:
		file_name = file_.split('/')[-1].replace('.xml', '')

		with open(file_, 'r') as reader:
			annot_dict.update({f"{file_name}" : xmltodict.parse(reader.read())})

		with open(json_path, 'w') as writer:
			writer.write(json.dumps(annot_dict))

		#print(f"Parsed xml to json of {file_name}")

def parse_folder_to_folder(xml_path, json_path):
	files = glob.glob(os.path.join(xml_path, '*.xml'))

	if not os.path.exists(json_path):
		os.mkdir(json_path)

	for file_ in files:
		file_json = os.path.join(json_path, file_.split('/')[-1].replace('xml', 'json'))
		annot_dict = {}

		with open(file_, 'r') as reader:
			annot_dict = xmltodict.parse(reader.read())

		with open(file_json, 'w') as writer:
			writer.write(json.dumps(annot_dict))

if __name__ == '__main__':
	coco_format = get_coco_json_format()

	for file_ in files:
		mask_pat

	parse_folder_to_file(sys.argv[1], sys.argv[2])