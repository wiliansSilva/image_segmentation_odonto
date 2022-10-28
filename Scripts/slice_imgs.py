from skimage import io, util
import glob
import os
import math
import random
import argparse
from DataAugmentation import DataAugmentation
from tqdm import tqdm

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--images_path', '-ip', required=True)
	parser.add_argument('--masks_path', '-mp', required=True)
	parser.add_argument('--save_imgs_path', '-sip', required=True)
	parser.add_argument('--save_masks_path', '-smp', required=True)
	parser.add_argument('--size', '-s', default=512)
	args = parser.parse_args()

	imgs_filepath = glob.glob(os.path.join(args.images_path, "*"))

	if not os.path.exists(args.save_imgs_path):
		os.makedirs(args.save_imgs_path)

	if not os.path.exists(args.save_masks_path):
		os.makedirs(args.save_masks_path)

	for img_path in tqdm(imgs_filepath):
		image_id = img_path.split('/')[-1].split('.')[0]

		img = io.imread(img_path, as_gray=True)

		masks_filepath = glob.glob(os.path.join(args.masks_path, image_id, '*'))

		classes = [ class_.split('/')[-1].replace('.png', '') for class_ in masks_filepath ]
		masks = [ io.imread(mask_filepath, as_gray=True) for mask_filepath in masks_filepath ]

		sliced_imgs = DataAugmentation.slice_img(img, args.size)
		sliced_masks = [ DataAugmentation.slice_img(mask, args.size) for mask in masks ]

		for i, sliced_img in enumerate(sliced_imgs):
			slice_id = image_id + "-" + str(i+1).zfill(2)
			try:
				io.imsave(os.path.join(args.save_imgs_path, slice_id + '.jpg'), util.img_as_ubyte(sliced_img), check_contrast=False)
			except FileExistsError:
				print(f"O arquivo {os.path.join(args.save_imgs_path, slice_id + '.jpg')} já existe!")
			
		for i, class_ in enumerate(sliced_masks):
			for j, mask in enumerate(class_):
				slice_id = image_id + "-" + str(j+1).zfill(2)
				if not os.path.exists(os.path.join(args.save_masks_path, slice_id)):
					os.makedirs(os.path.join(args.save_masks_path, slice_id))
				try:
					io.imsave(os.path.join(args.save_masks_path, slice_id, classes[i] + '.png'), util.img_as_ubyte(mask), check_contrast=False)
				except FileExistsError:
					print(f"O arquivo {os.path.join(args.save_masks_path, slice_id, classes[i] + '.png')} já existe!")