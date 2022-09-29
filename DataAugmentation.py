from skimage import io, transform, util
import numpy as np
import glob
import math
import matplotlib.pyplot as plt

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

	@classmethod
	def crop_img_into_imgs(cls, img:np.ndarray, size:tuple[int, int]) -> list[np.ndarray]:
		mult_x = img.shape[0] / size[0]
		mult_y = img.shape[1] / size[1]
		
		img_padded = cls.pad_img(img, (size[0] * math.ceil(mult_x), size[1] * math.ceil(mult_y)))

		imgs = []

		for i in range(math.ceil(mult_x)):
			for j in range(math.ceil(mult_y)):
				imgs.append(img_padded[(i * size[0]):((i + 1) * size[0]), (j * size[1]):((j + 1) * size[1])])
			
		return imgs

	@staticmethod
	def resize_img_mask(img:np.ndarray, masks:list, size:tuple[int, int]):
		resized_img = transform.resize(img, size)
		resized_masks_ = []
		for mask in masks[i]:
			resized_mask = transform.resize(mask, size)
			resized_masks.append(resized_mask)

		return resized_img, resized_masks_

	@classmethod
	def resize_imgs_masks(cls, imgs:list[np.ndarray], masks:list[list], size:tuple[int, int]):
		resized_imgs = []
		resized_masks = []

		for img, mask in zip(imgs, masks):
			resized_img, resized_masks_ = cls.resize_img_mask(img, mask, size)
			resized_imgs.append(resized_img)
			resized_masks.append(resized_masks_)

		return resized_imgs, resized_masks

	@staticmethod
	def rotate_img_mask(img:np.ndarray, masks:np.ndarray, angle:float):
		rotated_img = transform.rotate(img, angle)
		rotated_masks = transform.rotate(masks, angle)

		return rotated_img, rotated_masks

	@classmethod
	def rotate_imgs_masks(cls, imgs:list[np.ndarray], masks:list, angle:float):
		rotated_imgs = []
		rotated_masks = []

		for i, img in enumerate(imgs):
			rotated_img, rotated_mask = cls.rotate_img_mask(img, masks[i], angle)
			rotated_imgs.append(rotated_img)
			rotated_masks.append(rotated_mask)

		return rotated_imgs, rotated_masks

	@staticmethod
	def fliplr_img_mask(img:np.ndarray, mask:np.ndarray) -> tuple[np.ndarray, list]:
		flipped_img = np.fliplr(img)
		flipped_mask = np.fliplr(mask)
		
		return flipped_img, flipped_mask

	@classmethod
	def fliplr_imgs_masks(cls, imgs:list[np.ndarray], masks:list) -> tuple[list, list]:
		flipped_imgs = []
		flipped_masks = []
		for img, mask in zip(imgs, masks):
			flipped_img, flipped_masks_ = cls.fliplr_img_mask(img, mask)
			flipped_imgs.append(flipped_img)
			flipped_masks.append(flipped_masks_)

		return flipped_imgs, flipped_masks

	@staticmethod
	def random_noise_imgs_masks(imgs:list[np.ndarray], masks:list, mode='gaussian') -> tuple[list, list]:
		noised_imgs = []
		for img in imgs:
			noised_imgs.append(util.random_noise(img, mode))

		return noised_imgs, masks

if __name__ == '__main__':
	img = io.imread('./Images/Images-02/imagem-053.jpg', as_gray=True)
	imgs = DataAugmentation.crop_img_into_imgs(img, (256,256))
	for img_ in imgs:
		plt.imshow(img_)
		plt.waitforbuttonpress()