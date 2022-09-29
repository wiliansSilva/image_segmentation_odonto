from skimage import io, transform, util
import numpy as np
import glob
import os
import math
import matplotlib.pyplot as plt
import multiprocessing as mp
import threading
from threading import Thread
import random
import datetime
import segmentation_models as sm
import tensorflow as tf

class Dataset(tf.keras.utils.Sequence):
	def __init__(self, x_set, y_set, batch_size, size:tuple=(512,512)):
		self.x, self.y = x_set, y_set
		random.shuffle(self.x)
		self.batch_size = batch_size
		self.size = size

	def __len__(self):
		return math.ceil(len(self.x) / self.batch_size)

	def __getitem__(self, idx):
		batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
		#batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

		imgs = []
		masks = []

		for filename_img in batch_x:
			img = io.imread(filename_img, as_gray=True)
			cropped_imgs = crop_img_into_imgs(img, self.size)

			masks_ = [ [] for _ in range(len(cropped_imgs)) ]
			mask_path = filename_img.split('/')[-1].split('.')[0]
			mask_path = glob.glob(os.path.join(self.y, mask_path, '*'))

			for filename_mask in mask_path:
				mask = io.imread(filename_mask, as_gray=True)
				cropped_masks = crop_img_into_imgs(mask, self.size)

				for i, cropped_mask in enumerate(cropped_masks):
					masks_[i].append(cropped_mask)
				
			masks_ = [ np.array(mask).swapaxes(0, -1) for mask in masks_ ]

			imgs.extend(cropped_imgs)
			masks.extend(masks_)
		
		imgs = np.array(imgs)
		imgs = np.expand_dims(imgs, -1)
		print(imgs.shape)
		masks = np.array(masks)

		return imgs, masks

	def split(self, split_rate):
		splitted_x = self.x[:math.ceil(split_rate * len(self.x))]
		self.x = self.x[math.ceil(split_rate * len(self.x)):]

		return Dataset(splitted_x, self.y, math.ceil(split_rate * self.batch_size), self.size)

class DatasetTrainer:
	def __init__(self, imgs_path:str, masks_path:str, model, val_split:float=0.2, callbacks:list=None, epochs:int=3, batch_size=30, size:tuple[int, int]=(256,256), data_augmentation:bool=True, multiple_masks=True):
		self.imgs_location = glob.glob(os.path.join(imgs_path, '*'))
		random.shuffle(self.imgs_location)
		self.masks_location = masks_path
		self.model = model
		self.val_split = val_split
		self.callbacks = callbacks
		self.epochs = epochs
		self.batch_size = batch_size
		self.size = size
		self.data_augmentation = data_augmentation
		self.multiple_masks = multiple_masks

	def train(self):
		hist = []
		for training_period in range(math.ceil(len(self.imgs_location) / self.batch_size)):
			imgs_batch, masks_batch = load_data(self.imgs_location[(training_period * self.batch_size):((training_period + 1) * self.batch_size)], self.masks_location, self.size)
			#imgs_batch, masks_batch = augment_data(imgs_batch, masks_batch)
			# shuffle das lista de imagens e mascaras
			shuffled_list = list(zip(imgs_batch, masks_batch))
			random.shuffle(shuffled_list)
			imgs_batch, masks_batch = zip(*shuffled_list)
			imgs_batch = np.array(imgs_batch)
			imgs_batch = np.expand_dims(imgs_batch, -1)
			masks_batch = np.array(masks_batch)
			
			model.fit(imgs_batch, masks_batch, epochs=self.epochs, validation_split=self.val_split, callbacks=self.callbacks)

		return hist

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

if __name__ == '__main__':
	# sm.set_framework('tf.keras')
	# model = sm.Unet(classes=6, activation='softmax', input_shape=(None, None, 1), encoder_weights=None)
	# model.compile('Adam', loss=sm.losses.bce_jaccard_loss, metrics=[sm.metrics.iou_score])
	# horario = datetime.datetime.now()
	# callbacks = [tf.keras.callbacks.EarlyStopping(patience=3), tf.keras.callbacks.ModelCheckpoint(f'./Backups/Models/{horario.isoformat()}.hdf5', save_best_only=True)]

	# imgs_path = glob.glob('./Images/Images-02/*')
	# masks_path = './Masks/Masks-02/'

	# dataset = Dataset(imgs_path, masks_path, 1, size=(256,256))
	# validation_dataset = dataset.split(0.2)
	# # dataset = DatasetTrainer('./Images/Images-02/', './Masks/Masks-02/', model, batch_size=1, callbacks=callbacks, size=(256,256), data_augmentation=False)
	# # hist = dataset.train()

	# model.fit(x=dataset, epochs=10, validation_data=validation_dataset, callbacks=callbacks, use_multiprocessing=True)