import xml.etree.ElementTree as ET
import numpy as np
import cv2
import glob
import os
import argparse
from PIL import ImageColor, ImageDraw, Image

'''
map colors

tooth = white
restoration = blue
crown = yellow
dental implant = green
root canal treatment = purple
pulp = red - IGNORAR 
'''

# ANNOTATIONS = "Annotations-02--03-05-2022"
# IMAGES_PATH = "Images-02"
# OUTPUT_PATH = 'Masks-02'
# OUTPUT_COMBINED_PATH = 'Combined-Masks-02'
CLASSES = ['root canal treatment', 'restoration', 'crown', 'dental implant', 'tooth', 'pulp']
COLORS = ["green", "tomato", "blue", "yellow", "purple", "orange", "white"]

def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('annotations_path')
	parser.add_argument('images_path')
	parser.add_argument('masks_path')
	parser.add_argument('combined_path')
	return parser.parse_args()

def main(annotations_path, images_path, masks_path, combined_path):
	# get all paths to xmls in path
	files = glob.glob(os.path.join(annotations_path, "*.xml"))
	points = []
	# dic = {}

	for file in files:
		tree = ET.parse(file)
		root = tree.getroot()
        # read image to get shape
		image = cv2.imread(file.replace(annotations_path,images_path).replace("xml","jpg"))

		# create mask image with zeros
		try:
			# creating masks with zeros
			masks = [ np.zeros(image.shape) for class_ in CLASSES ]
			
			# creating a combined mask for visualization
			combined = np.zeros(image.shape)
			# loop to all objects in xml file
			i = 0
			for neighbor in root.findall("object"):
				# look if the polygon is deleted
				deleted = neighbor.find('deleted').text.strip()
				if deleted == '1':
					continue

				i += 1
				polygon = neighbor.find('polygon')
				# get points
				for pt in polygon.findall('pt'):
					x = int(round(float(pt.find("x").text)))
					y = int(round(float(pt.find("y").text)))
					points.append((x,y))

				# identifying the class of this polygon
				class_ = neighbor.find('name').text
				if class_.split(' ')[0] == 'tooth':
					class_ = 'tooth'

				if class_ in CLASSES:
					# getting the index of the class
					class_index = CLASSES.index(class_)
					color = 1
					#color = (255, 0, 0)
					
					r, g, b = ImageColor.getrgb(COLORS[class_index])
					
					# draw the masks
					masks[class_index] = cv2.fillPoly(masks[class_index], np.int32([points]), color)
					# draw the combined masks, if the class is tooth it will draw a contours
					if class_ == 'tooth':
						cv2.drawContours(combined, np.int32([points]), -1, (b, g, r), 5)
						# img = Image.fromarray(combined, mode='RGB')
						# ImageDraw.Draw(img).polygon(points, fill=None, outline=COLORS[class_index], width=5)
						# combined = np.array(img)
					else:
						combined = cv2.fillPoly(combined, np.int32([points]), (b, g, r))
				points = []

			# getting the identifier of the image
			image_id = file.split('/')[-1].replace('.xml', '')

			# if image_id in ANNOTATIONS_NAMES:
			# 	dic.update({image_id : i})

			# verifying the existence of some folders
			if not os.path.exists(masks_path):
				os.mkdir(masks_path)

			if not os.path.exists(combined_path):
				os.mkdir(combined_path)

			image_masks_path = os.path.join(masks_path, "{}".format(image_id.zfill(3))) # zfill completa oq falta da string com zeros
			if not os.path.exists(image_masks_path):
				os.mkdir(image_masks_path)

			# saving each class into a mask individually
			for i, mask in enumerate(masks):
				class_ = CLASSES[i]
				cv2.imwrite(os.path.join(image_masks_path, f"{class_}.png"), mask)

			# saving the combined mask
			cv2.imwrite(os.path.join(combined_path, "{}.png".format(image_id.zfill(3))), combined)

			print("Created mask from file: ", file)

		except AttributeError:
			print("Shape not found")

	# dic_sort = sorted(dic.keys())
	# for name in dic_sort:
	# 	print(f"{name} : {dic[name]}")

if __name__ == '__main__':
	args = parse_arguments()
	main(args.annotations_path, args.images_path, args.masks_path, args.combined_path)
