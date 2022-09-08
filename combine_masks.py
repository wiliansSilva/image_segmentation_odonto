import glob
from PIL import Image
import os
import numpy as np

# This is used to combine an image with the combined masks
# Important for visualizations

INPUT_COMBINED_PATH = './Combined-Masks-02/'
INPUT_IMGS_PATH = './Images-02/'
OUTPUT_COMBINED_MASKS_PATH = './combined_masks/'
OUTPUT_COMBINED_MASK_IMG_PATH = './Combined-Images-02/'

def combine_folder():
	img_paths = sorted(os.listdir(INPUT_IMGS_PATH))
	combined_paths = sorted(os.listdir(INPUT_COMBINED_PATH))

	if not os.path.exists(OUTPUT_COMBINED_MASK_IMG_PATH):
		os.mkdir(OUTPUT_COMBINED_MASK_IMG_PATH)

	for img_path, combined_path in zip(img_paths, combined_paths):
		combined_img = combine_masks_and_image(os.path.join(INPUT_IMGS_PATH, img_path), os.path.join(INPUT_COMBINED_PATH, combined_path))
		combined_img.save(os.path.join(OUTPUT_COMBINED_MASK_IMG_PATH, combined_path))
		print('Combined image generated: {}'.format(img_path.replace('.jpg', '')))

def combine_masks_and_image(img_path:str, combined_mask_path:str):
	combined = Image.open(img_path)
	combined = combined.convert("RGBA")

	masks = Image.open(combined_mask_path)
	masks = masks.convert("RGBA")
	combined_mask = Image.blend(combined, masks, 0.3)

	return combined_mask

if __name__ == '__main__':
	combine_folder()