import xmltodict
import json
import glob
import os
import sys

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
	#rename_xmls(sys.argv[1], sys.argv[2])
	parse_folder_to_file(sys.argv[1], sys.argv[2])