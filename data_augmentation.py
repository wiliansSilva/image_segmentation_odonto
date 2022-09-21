from skimage import io, transform, util
import numpy as np
import glob
import os
import math
import matplotlib.pyplot as plt
import multiprocessing as mp
import threading
from threading import Thread

IMGS_PATH = './Images/Images-02/*'
MASKS_PATH = './Masks/Masks-02/'
CLASSES = ['crown', 'dental implant', 'pulp', 'restoration', 'root canal treatment', 'tooth']

def load_data(imgs_path:str, masks_path:str, size:tuple):
	imgs_path = glob.glob(imgs_path)
	mask_path = os.listdir(masks_path)
	masks_dirs = [ os.path.join(masks_path, folder) for folder in mask_path ]
	imgs = []
	masks = []

	for i, img_file in enumerate(imgs_path):
		masks_ = []
		img = io.imread(img_file, as_gray=True)
		img = transform.resize(img, size)
		imgs.append(img)
		masks_names = glob.glob(os.path.join(masks_dirs[i], '*.png'))
		for j, mask_file in enumerate(masks_names):
			mask = io.imread(mask_file, as_gray=True)
			mask = transform.resize(mask, size)
			masks_.append(mask)
		masks.append(masks_)

	return imgs, masks

def pad_img(img:np.ndarray, size:tuple[int, int]) -> np.ndarray:
	mult_x = img.shape[0] / size[0]
	mult_y = img.shape[1] / size[1]

	img_padded = np.pad(img, ( (0, (math.ceil(mult_y) * size[1]) - img.shape[1]), (0, (math.ceil(mult_x) * size[0]) - img.shape[0]), (0,0) ), 'constant', constant_values=(0))

	return img_padded

def crop_img_into_imgs(img:np.ndarray, size:tuple[int, int]) -> list[np.ndarray]:
	width = img.shape[0]
	height = img.shape[1]
	mult_x = width / size[0]
	mult_y = height / size[1]
	
	img_padded = pad_img(img, (size[0] * math.ceil(mult_x), size[1] * math.ceil(mult_y)))

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

def rotate_img_mask(img:np.ndarray, masks:list, angle:float):
	rotated_img = transform.rotate(img, angle)
	rotated_masks = []
	for mask in masks:
		rotated_mask = transform.rotate(mask, angle)
		rotated_masks_.append(rotated_mask)
	rotated_masks.append(rotated_masks_)

	return rotated_img, rotated_masks

def rotate_imgs_masks(imgs:list[np.ndarray], masks:list[list], angle:float):
	rotated_imgs = []
	rotated_masks = []

	for i, img in enumerate(imgs):
		rotated_img, rotated_mask = rotate_img_mask(img, masks[i], angle)
		rotated_imgs.append(rotated_img)
		rotated_masks.append(rotated_mask)

	return rotated_imgs, rotated_masks

def fliplr_img_mask(img:np.ndarray, mask:list[np.ndarray]) -> tuple[np.ndarray, list]:
	flipped_img = np.fliplr(img)
	flipped_masks_ = []
	for mask in masks:
		flipped_mask = np.fliplr(mask)
		flipped_masks_.append(flipped_mask)
	
	return flipped_img, flipped_masks_

def fliplr_imgs_masks(imgs:list[np.ndarray], masks:list[list]) -> tuple[list, list]:
	flipped_imgs = []
	flipped_masks = []
	for img, mask in zip(imgs, masks):
		flipped_img, flipped_masks_ = fliplr_img_mask(img, mask)
		flipped_imgs.append(flipped_img)
		flipped_masks.append(flipped_masks_)

	return flipped_imgs, flipped_masks

def random_noise_imgs_masks(imgs:list[np.ndarray], masks:list[list], mode='gaussian') -> tuple[list, list]:
	noised_imgs = []
	for img in imgs:
		noised_img.append(util.random_noise(img, mode))

	return noised_imgs, masks

def save_img_mask(index:int, img:np.ndarray, mask:list, imgs_path:str, masks_path:str):
	io.imsave(os.path.join(imgs_path, f'imagem-{str(index).zfill(3)}.jpg'), img)
	for j, mask in enumerate(masks_):
		io.imsave(os.path.join(masks_path, f'imagem-{str(index).zfill(3)}', f'{CLASSES[j]}.png'), mask)

def save_imgs_masks(imgs:list, masks:list, imgs_path:str, masks_path:str):
	if not os.path.exists(imgs_path):
		os.mkdir(imgs_path)
	if not os.path.exists(masks_path):
		os.mkdir(masks_path)

	threads = []
	for i, (img, masks_) in enumerate(zip(imgs, masks)):
		t = Thread(target=save_img_mask, args=(i, img, masks_, imgs_path, masks_path))
		t.start()
		threads.append(t)

	for t in threads:	
		t.join()

def augment_data(imgs:list[np.ndarray], masks:list[list]):
	pass

if __name__ == '__main__':
	imgs, masks = load_data(IMGS_PATH, MASKS_PATH, (512,512))
	#resized_imgs, resized_masks = resize_imgs_masks(imgs, masks, (512,512))
	save_imgs_masks(imgs, masks, './Images/Resized/512', './Masks/Resized/512')
