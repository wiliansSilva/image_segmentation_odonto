from skimage import io, util
import glob
import os
import math
import random
import argparse
from DataAugmentation import DataAugmentation
from tqdm import tqdm

'''
def augmentation_routine(img, masks):
	augmented_imgs = []
	augmented_masks = []

	aug_img = DataAugmentation.random_noise_img_mask(img)
	augmented_imgs.append(aug_img)
	augmented_masks.append(masks)

	return augmented_imgs, augmented_masks

'''
def augmentation_routine(img, masks):
	augmented_imgs = []
	augmented_masks = []

	aug_img, aug_mask = DataAugmentation.fliplr_img_mask(img, masks)
	augmented_imgs.append(aug_img)
	augmented_masks.append(aug_mask)

	aug_img, aug_mask = DataAugmentation.flipud_img_mask(img, masks)
	augmented_imgs.append(aug_img)
	augmented_masks.append(aug_mask)

	for angle in [45, 90, 135, 180, 225, 270, 315]:
		aug_img, aug_mask = DataAugmentation.rotate_img_mask(img, masks, angle)
		augmented_imgs.append(aug_img)
		augmented_masks.append(aug_mask)

	return augmented_imgs, augmented_masks


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--images-path', '-ip', required=True)
	parser.add_argument('--masks-path', '-mp', required=True)
	parser.add_argument('--save_imgs_path', '-sip', required=True)
	parser.add_argument('--save_masks_path', '-smp', required=True)
	args = parser.parse_args()

	if not os.path.exists(args.save_imgs_path):
		os.makedirs(args.save_imgs_path)

	if not os.path.exists(args.save_masks_path):
		os.makedirs(args.save_masks_path)

	imgs_filepath = glob.glob(os.path.join(args.images_path, "*"))

	for img_path in tqdm(imgs_filepath):
		image_id = img_path.split('/')[-1].split('.')[0]

		img = io.imread(img_path, as_gray=True)

		masks_filepath = glob.glob(os.path.join(args.masks_path, image_id, '*'))

		classes = [ class_.split('/')[-1].replace('.png', '') for class_ in masks_filepath ]
		masks = [ io.imread(mask_filepath, as_gray=True) for mask_filepath in masks_filepath ]

		augmented_imgs, augmented_masks = augmentation_routine(img, masks)

		for i, aug_img in enumerate(augmented_imgs):
			aug_id = image_id + "-" + str(i+1).zfill(2)
			try:
				io.imsave(os.path.join(args.save_imgs_path, aug_id + '.jpg'), util.img_as_ubyte(aug_img), check_contrast=False)
			except FileExistsError:
				print(f"O arquivo {os.path.join(args.save_imgs_path, aug_id + '.jpg')} já existe!")
			
		for i, aug_masks in enumerate(augmented_masks):
			aug_id = image_id + "-" + str(i+1).zfill(2)
			if not os.path.exists(os.path.join(args.save_masks_path, aug_id)):
				os.makedirs(os.path.join(args.save_masks_path, aug_id))
			for j, aug_mask in enumerate(aug_masks):
				try:
					io.imsave(os.path.join(args.save_masks_path, aug_id, classes[j] + '.png'), util.img_as_ubyte(aug_mask), check_contrast=False)
				except FileExistsError:
					print(f"O arquivo {os.path.join(args.save_masks_path, aug_id, classes[j] + '.png')} já existe!")