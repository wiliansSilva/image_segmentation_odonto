import xml.etree.ElementTree as ET
import numpy as np
import cv2
from skimage import io, util
import glob
import os
import sys
import argparse
from PIL import ImageColor, ImageDraw, Image
import matplotlib.pyplot as plt

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

ANNOTATIONS_PATH = "../Annotations/annotations"
IMAGES_PATH = "../Images/images"
MASKS_PATH = "../Masks/cropped_masks"
SIZE = ""
CROPPED_IMAGES_PATH = "../Images/cropped_images"
CROPPED_COMBINED_PATH = "../Masks/combined/cropped_masks"

PAD = 20 # pixels

def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('annotations_path')
	parser.add_argument('images_path')
	parser.add_argument('masks_path')
	parser.add_argument('combined_path')
	parser.add_argument('size')
	return parser.parse_args()

def convert_single_annotation_object(size, annotations_path, images_path, masks_path, croped_imgs_path):
	annotations_filenames = glob.glob(os.path.join(annotations_path, "*.xml"))

	for annotations_filename in annotations_filenames:
		tree = ET.parse(annotations_filename)
		root = tree.getroot()

		img = io.imread(annotations_filename.replace(annotations_path, images_path).replace("xml", "jpg"), as_gray=True)

		try:
			masks = [ [] for _ in CLASSES ]
			imgs = [ [] for _ in CLASSES ]

			for neighbor in root.findall("object"):
				deleted = neighbor.find("deleted").text.strip()
				if deleted == "1":
					continue

				points = []
				mask = np.zeros((size[0], size[1]), dtype=np.uint8)
				polygon = neighbor.find("polygon")
				min_x = None
				min_y = None
				max_x = None
				max_y = None
				x_s = []
				y_s = []

				for pt in polygon.findall("pt"):
					x = int(round(float(pt.find("x").text)))
					y = int(round(float(pt.find("y").text)))
					if min_x > x or min_x == None:
						min_x = x
					if min_y > y or min_y == None:
						min_y = y
					if max_x > x or max_x == None:
						max_x = x
					if max_y > y or max_y == None:
						max_y = y
					x_s.append(x)
					y_s.append(y)

				for x, y in zip(x_s, y_s):
					x -= min_x
					y -= min_y
					points.append((x, y))

				class_ = neighbor.find("name").text.strip().replace("_", "")
				if class_[:5] == "tooth":
					class_ = "tooth"
				elif class_ == "implant":
					class_ = "dental implant"

				if class_ in CLASSES:
					class_index = CLASSES.index(class_)

					masks[class_index].append(cv2.fillPoly(mask, np.int32([points]), 1))
					imgs[class_index].append(img[min_x:max_x, min_y:max_y])

				if not os.path.exists(masks_path):
					os.makedirs(masks_path)

				if not os.path.exists(croped_imgs_path):
					os.makedirs(croped_imgs_path)

				image_id = annotations_filename.split('/')[-1].replace('.xml', '')

				for i in range(len(masks)):
					class_ = CLASSES[i]
					for j in range(len(masks[i])):
						if not os.path.exists(os.path.join(masks_path, f"{image_id}-{j}")):
							os.makedirs(os.path.join(masks_path, f"{image_id}-{j}"))
						io.imsave(os.path.join(masks_path, f"{image_id}-{j}", f"{class_}.png"), masks[i][j], check_contrast=False)
						io.imsave(os.path.join(croped_imgs_path, f"{image_id}-{j}.jpg"), imgs[i][j], check_contrast=False)

		except:
			print(f"Erro com o arquivo: {annotations_filename}")

def convert_entire_mask(annotations_path, images_path, masks_path):
	annotations_filenames = glob.glob(os.path.join(annotations_path, '*'))

	for annotation_filename in annotations_filenames:
		image_id_str = annotation_filename.split("/")[-1].split(".")[0]
		
		tree = ET.parse(annotations_filename)
		root = tree.getroot()

		img = io.imread(os.path.join(images_path, f"{image_id_str}.jpg"), as_gray=True)

		mask = np.zeros((img.shape[0], img.shape[1], len(CLASSES)), dtype=np.uint8)

		if not os.path.exists(masks_path):
			os.makedirs(masks_path)

		if not os.path.exists(croped_imgs_path):
			os.makedirs(croped_imgs_path)

		for neighbor in root.findall("object"):
			deleted = neighbor.find("deleted").text.strip()
			if deleted == "1":
				continue

			polygon = neighbor.find("polygon")
			points = []
			for pt in polygon.findall("pt"):
				x = int(round(float(pt.find("x"))))
				y = int(round(float(pt.find("y"))))
				points.append((x, y))

			class_ = neighbor.find("name").text.strip().replace("_", "")
			if class_[:5] == "tooth":
				class_ = "tooth"
			elif class_ == "implant":
				class_ = "dental implant"

			if class_ in CLASSES:
				class_index = CLASSES.index(class_)
				cv2.fillPoly(mask[:,:,class_index], np.int32([points]), 1)

		for i, class_ in enumerate(CLASSES):
			if not os.path.exists(os.path.join(masks_path, f"{image_id_str}")):
				os.makedirs(os.path.join(os.path.join(masks_path, f"{image_id_str}")))
			io.imsave(os.path.join(masks_path, f"{image_id_str}", f"{class_}.png"), mask[:,:,i], check_contrast=False)
			
def convert_to_object(annotations_path, images_path, cropped_images_path, masks_path, cropped_combined_path):
	annotations_filenames = glob.glob(os.path.join(annotations_path, '*'))
	
	for annotation_filename in annotations_filenames:
		# image-xxx
		image_id = annotation_filename.split("/")[-1].split(".")[0]
		
		tree = ET.parse(annotation_filename)
		root = tree.getroot()

		img = io.imread(os.path.join(images_path, f"{image_id}.jpg"), as_gray=True)

		if not os.path.exists(masks_path):
			os.makedirs(masks_path)

		if not os.path.exists(cropped_images_path):
			os.makedirs(cropped_images_path)

		if not os.path.exists(cropped_combined_path):
			os.makedirs(cropped_combined_path)

		rectangles = []
		mask = [ np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8) for _ in CLASSES ]
		combined = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

		for neighbor in root.findall("object"):
			deleted = neighbor.find("deleted").text.strip()
			if deleted == "1":
				continue

			polygon = neighbor.find("polygon")

			x_s = []
			y_s = []

			for pt in polygon.findall("pt"):
				x = int(round(float(pt.find("x").text)))
				y = int(round(float(pt.find("y").text)))
				if x < 0:
					x = 0
				if y < 0:
					y = 0

				x_s.append(x)
				y_s.append(y)

			if len(x_s) == 0:
				continue
			if len(y_s) == 0:
				continue

			points = list(zip(x_s, y_s))

			class_ = neighbor.find("name").text.strip().replace("_", "")
			if class_[:5] == "tooth":
				class_ = "tooth"
			elif class_ == "implant":
				class_ = "dental implant"

			if class_ == "tooth":
				x_s = np.array(x_s)
				y_s = np.array(y_s)

				min_x = np.min(x_s)
				max_x = np.max(x_s)

				min_y = np.min(y_s)
				max_y = np.max(y_s)
				rectangles.append((min_x, max_x, min_y, max_y))

			if class_ in CLASSES:
				class_index = CLASSES.index(class_)
				cv2.fillPoly(mask[class_index], np.int32([points]), 1)
				
				r, g, b = ImageColor.getrgb(COLORS[class_index])
			
				# draw the combined masks, if the class is tooth it will draw a contours
				if class_ == 'tooth':
					cv2.drawContours(combined, np.int32([points]), -1, (b, g, r), 5)
				else:
					combined = cv2.fillPoly(combined, np.int32([points]), (b, g, r))

		for i, rectangle in enumerate(rectangles):
			i += 1
			if len(rectangle) != 4:
				continue
			if rectangle[0] == rectangle[1] or rectangle[2] == rectangle[3]:
				continue
			if not os.path.exists(os.path.join(masks_path, f"{image_id}-{str(i).zfill(2)}")):
				os.makedirs(os.path.join(os.path.join(masks_path, f"{image_id}-{str(i).zfill(2)}")))
			print(rectangle)
			print(image_id)
			print(img.shape)
			try:
				io.imsave(os.path.join(cropped_images_path, f"{image_id}-{str(i).zfill(2)}.jpg"), img[rectangle[2]:rectangle[3], rectangle[0]:rectangle[1]], check_contrast=False)
			except:
				plt.imshow(img)
				plt.show()
			io.imsave(os.path.join(cropped_combined_path, f"{image_id}-{str(i).zfill(2)}.png"), combined[rectangle[2]:rectangle[3], rectangle[0]:rectangle[1]], check_contrast=False)
			for j, class_id in enumerate(CLASSES):
				io.imsave(os.path.join(masks_path, f"{image_id}-{str(i).zfill(2)}", f"{class_id}.png"), mask[j][rectangle[2]:rectangle[3], rectangle[0]:rectangle[1]], check_contrast=False)

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
			masks = [ np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8) for class_ in CLASSES ]
			
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
				if class_[:5] == 'tooth':
					class_ = 'tooth'
				elif class_ == 'implant':
					class_ = 'dental implant'

				if class_ in CLASSES:
					# getting the index of the class
					class_index = CLASSES.index(class_)
					color = 1
					
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
				os.makedirs(masks_path)

			if not os.path.exists(combined_path):
				os.makedirs(combined_path)

			image_masks_path = os.path.join(masks_path, f"{image_id.zfill(3)}") # zfill completa oq falta da string com zeros
			
			if not os.path.exists(image_masks_path):
				os.makedirs(image_masks_path)

			# saving each class into a mask individually
			for i, mask in enumerate(masks):
				class_ = CLASSES[i]
				#io.imsave(os.path.join(image_masks_path, f"{class_}.png"), mask, check_contrast=False)
				cv2.imwrite(os.path.join(image_masks_path, f"{class_}.png"), mask)

			# saving the combined mask
			cv2.imwrite(os.path.join(combined_path, f"{image_id.zfill(3)}.png"), combined)

			print("Created mask from file: ", file)

		except AttributeError:
			print(f"Erro com a anotação {file}")
			
if __name__ == '__main__':
	#main()
	#args = parse_arguments()
	#main(args.annotations_path, args.images_path, args.masks_path, args.combined_path)
	#convert_single_annotation_object((256,256), sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
	#convert_entire_mask(args.annotations_path, args.images_path, args.masks_path)
	convert_to_object(ANNOTATIONS_PATH, IMAGES_PATH, CROPPED_IMAGES_PATH, MASKS_PATH, CROPPED_COMBINED_PATH)
