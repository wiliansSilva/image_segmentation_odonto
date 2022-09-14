import glob
from PIL import Image
import os
import numpy as np
import argparse

# This is used to combine an image with the combined masks
# Important for visualizations

# INPUT_COMBINED_PATH = './Combined-Masks-02/'
# INPUT_IMGS_PATH = './Images-02/'
# OUTPUT_COMBINED_MASK_IMG_PATH = './Combined-Images-02/'

def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('combined_masks_path')
	parser.add_argument('images_path')
	parser.add_argument('combined_mask_and_img_path')
	return parser.parse_args()

def combine_folder(combined_masks_path, images_path, combined_mask_and_img_path):
	img_paths = sorted(os.listdir(images_path))
	combined_paths = sorted(os.listdir(combined_masks_path))

	if not os.path.exists(combined_mask_and_img_path):
		os.mkdir(combined_mask_and_img_path)

	for img_path, combined_path in zip(img_paths, combined_paths):
		combined_img = combine_masks_and_image(os.path.join(images_path, img_path), os.path.join(combined_masks_path, combined_path))
		combined_img.save(os.path.join(combined_mask_and_img_path, combined_path))
		print('Combined image generated: {}'.format(img_path.replace('.jpg', '')))

def combine_masks_and_image(img_path:str, combined_mask_path:str):
	combined = Image.open(img_path)
	combined = combined.convert("RGBA")

	masks = Image.open(combined_mask_path)
	masks = masks.convert("RGBA")
	combined_mask = Image.blend(combined, masks, 0.3)

	return combined_mask

if __name__ == '__main__':
	args = parse_arguments()
	combine_folder(args.combined_masks_path, args.images_path, args.combined_mask_and_img_path)