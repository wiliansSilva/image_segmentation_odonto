from skimage import io, transform, util
import numpy as np
import glob
import os
import math
import matplotlib.pyplot as plt
import random
import datetime

IMGS_PATH = './Images/Images-02/*'
MASKS_PATH = './Masks/Masks-02/'
CLASSES = ['crown', 'dental implant', 'pulp', 'restoration', 'root canal treatment', 'tooth']

def augment_data(imgs:list[np.ndarray], masks:list[np.ndarray]):
		rotated_imgs = []
		rotated_masks = []
		for angle in [90, 180, 270]:
			x, y = rotate_imgs_masks(imgs, masks, angle)
			rotated_imgs.extend(x)
			rotated_masks.extend(y)
		
		flipped_imgs, flipped_masks = fliplr_imgs_masks(imgs, masks)

		noised_imgs, noised_masks = random_noise_imgs_masks(imgs, masks)

		imgs.extend(rotated_imgs)
		masks.extend(rotated_masks)

		imgs.extend(flipped_imgs)
		masks.extend(flipped_masks)

		imgs.extend(noised_imgs)
		masks.extend(noised_masks)

		return imgs, masks

def load_img_and_masks_crop(img_location, masks_location, size=(512,512)):
	id = img_location.split('/')[-1].split('.')[0]
	masks_location = glob.glob(os.path.join(masks_location, id, '*'))

	img = io.imread(img_location, as_gray=True)
	imgs = crop_img_into_imgs(img, size)
	masks = [ [] for _ in range(len(imgs)) ]

	for mask_location in masks_location:
		mask = io.imread(mask_location, as_gray=True)
		cropped_masks = crop_img_into_imgs(mask, size)

		for i, cropped_mask in enumerate(cropped_masks):
			masks[i].append(cropped_mask)
			
	masks = [ np.array(mask).swapaxes(0, -1) for mask in masks ]	

	return imgs, masks

def load_data(imgs_path:list, masks_path:str, size:tuple):
	imgs = []
	masks = []

	for img_path in imgs_path:
		x, y = load_img_and_masks_crop(img_path, masks_path, size)
		imgs.extend(x)
		masks.extend(y)

	return imgs, masks

def pad_img(img:np.ndarray, size:tuple[int, int]) -> np.ndarray:
	img_padded = np.pad(img, ( (0, size[0] - img.shape[0]), (0, size[1] - img.shape[1])), 'constant', constant_values=(0))
	return img_padded

def crop_img_into_imgs(img:np.ndarray, size:tuple[int, int]) -> list[np.ndarray]:
	print(img.shape)
	mult_x = img.shape[0] / size[0]
	mult_y = img.shape[1] / size[1]

	print(mult_x)
	print(mult_y)
	
	img_padded = pad_img(img, (size[0] * math.ceil(mult_x), size[1] * math.ceil(mult_y)))

	print(img_padded.shape)

	imgs = []

	for i in range(math.ceil(mult_x)):
		for j in range(math.ceil(mult_y)):
			imgs.append(img_padded[(i * size[0]):((i + 1) * size[0]), (j * size[1]):((j + 1) * size[1])])
		
	return imgs

def resize_img_mask(img:np.ndarray, masks:list, size:tuple[int, int]):
	resized_img = transform.resize(img, size)
	resized_masks_ = []
	for mask in masks[i]:
		resized_mask = transform.resize(mask, size)
		resized_masks.append(resized_mask)

	return resized_img, resized_masks_

def resize_imgs_masks(imgs:list[np.ndarray], masks:list[list], size:tuple[int, int]):
	resized_imgs = []
	resized_masks = []

	for img, mask in zip(imgs, masks):
		resized_img, resized_masks_ = resize_img_mask(img, mask, size)
		resized_imgs.append(resized_img)
		resized_masks.append(resized_masks_)

	return resized_imgs, resized_masks

def rotate_img_mask(img:np.ndarray, masks:np.ndarray, angle:float):
	rotated_img = transform.rotate(img, angle)
	rotated_masks = transform.rotate(masks, angle)

	return rotated_img, rotated_masks

def rotate_imgs_masks(imgs:list[np.ndarray], masks:list, angle:float):
	rotated_imgs = []
	rotated_masks = []

	for i, img in enumerate(imgs):
		rotated_img, rotated_mask = rotate_img_mask(img, masks[i], angle)
		rotated_imgs.append(rotated_img)
		rotated_masks.append(rotated_mask)

	return rotated_imgs, rotated_masks

def fliplr_img_mask(img:np.ndarray, mask:np.ndarray) -> tuple[np.ndarray, list]:
	flipped_img = np.fliplr(img)
	flipped_mask = np.fliplr(mask)
	
	return flipped_img, flipped_mask

def fliplr_imgs_masks(imgs:list[np.ndarray], masks:list) -> tuple[list, list]:
	flipped_imgs = []
	flipped_masks = []
	for img, mask in zip(imgs, masks):
		flipped_img, flipped_masks_ = fliplr_img_mask(img, mask)
		flipped_imgs.append(flipped_img)
		flipped_masks.append(flipped_masks_)

	return flipped_imgs, flipped_masks

def random_noise_imgs_masks(imgs:list[np.ndarray], masks:list, mode='gaussian') -> tuple[list, list]:
	noised_imgs = []
	for img in imgs:
		noised_imgs.append(util.random_noise(img, mode))

	return noised_imgs, masks

if __name__ == '__main__':
	img = io.imread('./Images/Images-02/imagem-053.jpg', as_gray=True)
	print(img.shape)
	imgs = crop_img_into_imgs(img, (512,512))
	#for img_ in imgs:
	#	plt.imshow(img_)
	#	plt.waitforbuttonpress()
	
	imgs_path = glob.glob(IMGS_PATH)

	imgs, masks = load_data(imgs_path, MASKS_PATH, (512,512))
	x = np.array(imgs)
	y = np.array(masks)
