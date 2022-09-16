import xmltodict
import json
import glob
import os
import sys

def parse_folder_to_file(xml_path, json_path):
	files = glob.glob(os.path.join(xml_path, '*.xml'))
	annot_dict = {}

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
	parse_folder_to_file(sys.argv[1], sys.argv[2])