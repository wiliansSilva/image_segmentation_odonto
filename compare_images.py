import matplotlib.pyplot as plt
from skimage import io
from PIL import Image
import numpy as np
import glob
import os

# Used to compare two diferent images combined with masks

PATH1 = 'Combined-Images-02/'
PATH2 = 'dentalxray/outputCombined/'

def show_window(img1:np.ndarray, title1:str, img2:np.ndarray, title2:str):
	fig = plt.figure()
	ax = fig.add_subplot(1, 2, 1)
	imgplot = plt.imshow(img1)
	ax.set_title(title1)
	ax = fig.add_subplot(1, 2, 2)
	imgplot = plt.imshow(img2)
	ax.set_title(title2)
	plt.waitforbuttonpress()
	plt.close()

def main():
	imgs_names1 = os.listdir(PATH1)
	imgs_names2 = os.listdir(PATH2)

	for img_name in imgs_names1:
		try:
			img1 = io.imread(os.path.join(PATH1, img_name))
			img2 = io.imread(os.path.join(PATH2, img_name))
			show_window(img1, os.path.join(PATH1, img_name), img2, os.path.join(PATH2, img_name))
			del(img1)
			del(img2)
		except Exception as ex:
			print(ex)
#142
if __name__ == '__main__':
	main()