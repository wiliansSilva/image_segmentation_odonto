from skimage import io, util
import glob
import os
import math
import random
import argparse
from DataAugmentation import DataAugmentation
from tqdm import tqdm

IMAGES_PATH = "../Images/cropped_images"
MASKS_PATH = "../Masks/cropped_masks"
SPLITED_IMGS_PATH = "../Images/splitted_cropped_images"
SPLITED_MASKS_PATH = "../Masks/splitted_cropped_masks"
SIZE = 256

def main(images_path, masks_path, splited_imgs_path, splited_masks_path, size):
	imgs_filepath = glob.glob(os.path.join(images_path, "*"))

	if not os.path.exists(splited_masks_path):
		os.makedirs(splited_masks_path)

	if not os.path.exists(splited_imgs_path):
		os.makedirs(splited_imgs_path)

	for img_path in tqdm(imgs_filepath):
		image_id = img_path.split('/')[-1].split('.')[0]

		img = io.imread(img_path, as_gray=True)

		masks_filepath = glob.glob(os.path.join(masks_path, image_id, '*'))

		classes = [ class_.split('/')[-1].replace('.png', '') for class_ in masks_filepath ]
		masks = [ DataAugmentation.pad_img(io.imread(mask_filepath, as_gray=True), (size, size)) for mask_filepath in masks_filepath ]

		padded_img = DataAugmentation.pad_img(img, (size, size))

		sliced_imgs = DataAugmentation.split_img(padded_img, size)
		sliced_masks = [ DataAugmentation.split_img(mask, size) for mask in masks ]

		for i, sliced_img in enumerate(sliced_imgs):
			slice_id = image_id + "-" + str(i+1).zfill(2)
			if sliced_img.shape[0] != size or sliced_img.shape[1] != size:
				continue
			try:
				io.imsave(os.path.join(splited_imgs_path, slice_id + '.jpg'), util.img_as_ubyte(sliced_img), check_contrast=False)
			except FileExistsError:
				print(f"O arquivo {os.path.join(splited_imgs_path, slice_id + '.jpg')} já existe!")
			
		for i, class_ in enumerate(sliced_masks):
			for j, mask in enumerate(class_):
				slice_id = image_id + "-" + str(j+1).zfill(2)
				if not os.path.exists(os.path.join(splited_masks_path, slice_id)):
					os.makedirs(os.path.join(splited_masks_path, slice_id))
				try:
					io.imsave(os.path.join(splited_masks_path, slice_id, classes[i] + '.png'), util.img_as_ubyte(mask), check_contrast=False)
				except FileExistsError:
					print(f"O arquivo {os.path.join(splited_masks_path, slice_id, classes[i] + '.png')} já existe!")

if __name__ == '__main__':
	main(IMAGES_PATH, MASKS_PATH, SPLITED_IMGS_PATH, SPLITED_MASKS_PATH, SIZE)
