from skimage import io, transform, util
import numpy as np
import glob
import math
import random
import argparse

class DataAugmentation():
	@staticmethod
	def pad_img(img:np.ndarray, size:tuple[int, int]) -> np.ndarray:
		img_padded = None

		# verifica se a imagem tem somente um canal
		if len(img.shape) == 2:
			img_padded = np.pad(img, ( (0, size[0] - img.shape[0]), (0, size[1] - img.shape[1])), 'constant', constant_values=(0))

		# verifica se a imagem tem mais de um canal
		elif len(img.shape) == 3:
			img_padded = np.pad(img, ( (0, size[0] - img.shape[0]), (0, size[1] - img.shape[1]), (0,0) ), 'constant', constant_values=(0))
		
		return img_padded

	@staticmethod
	def slice_img(img:np.ndarray, size:int) -> list[np.ndarray]:
		mult_x = math.ceil(img.shape[0] / size)
		mult_y = math.ceil(img.shape[1] / size)
		
		imgs = []

		for i in range(mult_x):
			for j in range(mult_y):
				if (i == mult_x - 1) and (j == mult_y - 1):
					imgs.append(img[(img.shape[0] - size):(img.shape[0]), (img.shape[1] - size):(img.shape[1])])
				elif j == mult_y - 1:
					imgs.append(img[(i * size):((i + 1) * size), (img.shape[1] - size):(img.shape[1])])
				elif i == mult_x - 1:
					imgs.append(img[(img.shape[0] - size):(img.shape[0]), (j * size):((j + 1) * size)])
				else:
					imgs.append(img[(i * size):((i + 1) * size), (j * size):((j + 1) * size)])
			
		return imgs

	@staticmethod
	def resize_img_mask(img:np.ndarray, masks:list, size:tuple[int, int]):
		resized_img = transform.resize(img, size)
		resized_masks_ = []
		for mask in masks[i]:
			resized_mask = transform.resize(mask, size)
			resized_masks.append(resized_mask)

		return resized_img, resized_masks_

	@staticmethod
	def rotate_img_mask(img:np.ndarray, masks:list, angle:float):
		rotated_img = transform.rotate(img, angle)
		rotated_masks = [ transform.rotate(mask, angle, mode='wrap') for mask in masks ]

		return rotated_img, rotated_masks

	@staticmethod
	def fliplr_img_mask(img:np.ndarray, masks:list) -> tuple[np.ndarray, list]:
		flipped_img = np.fliplr(img)
		flipped_masks = [ np.fliplr(mask) for mask in masks ]
		
		return flipped_img, flipped_masks

	@staticmethod
	def flipud_img_mask(img:np.ndarray, masks:list) -> tuple[np.ndarray, list]:
		flipped_img = np.flipud(img)
		flipped_masks = [ np.flipud(mask) for mask in masks ]
		
		return flipped_img, flipped_masks

	@staticmethod
	def random_noise_img_mask(img:np.ndarray, mode='gaussian') -> np.array:
		return util.random_noise(img, mode)

	@staticmethod
	def random_noise_imgs_masks(imgs:list[np.ndarray], masks:list, mode='gaussian') -> tuple[list, list]:
		noised_imgs = [ util.random_noise(img, mode) for img in imgs ]

		return noised_imgs, masks

	@staticmethod
	def random_darkness(img) -> np.array:
		return exposure.adjust_gamma(img, random.uniform(1.01, 1.5))

	@staticmethod
	def random_brightness(img) -> np.array:
		return exposure.adjust_gamma(img, random.uniform(0.5, 0.99))

	#####

	@classmethod
	def crop_img_into_imgs_with_pad(cls, img:np.ndarray, size:tuple[int, int]) -> list[np.ndarray]:
		mult_x = img.shape[0] / size[0]
		mult_y = img.shape[1] / size[1]
		
		img_padded = cls.pad_img(img, (size[0] * math.ceil(mult_x), size[1] * math.ceil(mult_y)))

		imgs = []

		for i in range(math.ceil(mult_x)):
			for j in range(math.ceil(mult_y)):
				imgs.append(img_padded[(i * size[0]):((i + 1) * size[0]), (j * size[1]):((j + 1) * size[1])])
			
		return imgs

	@classmethod
	def resize_imgs_masks(cls, imgs:list[np.ndarray], masks:list[list], size:tuple[int, int]):
		resized_imgs = []
		resized_masks = []

		for img, mask in zip(imgs, masks):
			resized_img, resized_masks_ = cls.resize_img_mask(img, mask, size)
			resized_imgs.append(resized_img)
			resized_masks.append(resized_masks_)

		return resized_imgs, resized_masks

	@classmethod
	def rotate_imgs_masks(cls, imgs:list[np.ndarray], masks:list, angle:float):
		rotated_imgs = []
		rotated_masks = []

		for i, img in enumerate(imgs):
			rotated_img, rotated_mask = cls.rotate_img_mask(img, masks[i], angle)
			rotated_imgs.append(rotated_img)
			rotated_masks.append(rotated_mask)

		return rotated_imgs, rotated_masks

	@classmethod
	def fliplr_imgs_masks(cls, imgs:list[np.ndarray], masks:list) -> tuple[list, list]:
		flipped_imgs = []
		flipped_masks = []
		for img, mask in zip(imgs, masks):
			flipped_img, flipped_masks_ = cls.fliplr_img_mask(img, mask)
			flipped_imgs.append(flipped_img)
			flipped_masks.append(flipped_masks_)

		return flipped_imgs, flipped_masks