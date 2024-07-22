from DataAugmentation import DataAugmentation
from skimage import io, transform

class DataLoader():
	@staticmethod
	def load_img_and_masks_resize(img_location:str, masks_location:str, img_as_gray:bool=True, size:tuple=(512,512)) -> tuple[np.array, np.array]:
		id = img_location.split('/')[-1].split('.')[0]
		masks_location = glob.glob(os.path.join(masks_location, id, '*'))

		img = io.imread(img_location, as_gray=img_as_gray)
		img = transform.resize(img, size)
		
		masks = []

		for mask_location in masks_location:
			mask = io.imread(mask_location, as_gray=True)
			masks.append(transform.resize(size))
				
		masks = np.array(masks)
		masks = masks.swapaxes(0, -1)

		return imgs, masks

	@staticmethod
	def load_img_and_masks_crop(img_location:str, masks_location:str, img_as_gray:bool=True, size=(512,512)):
		id = img_location.split('/')[-1].split('.')[0]
		masks_location = glob.glob(os.path.join(masks_location, id, '*'))

		img = io.imread(img_location, as_gray=img_as_gray)
		imgs = DataAugmentation.crop_img_into_imgs(img, size)

		# cria varias lista para ter as mascaras de cada recorte da imagem
		masks = [ [] for _ in range(len(imgs)) ]

		for mask_location in masks_location:
			mask = io.imread(mask_location, as_gray=True)
			cropped_masks = DataAugmentation.crop_img_into_imgs(mask, size)

			for i, cropped_mask in enumerate(cropped_masks):
				masks[i].append(cropped_mask)
				
		masks = [ np.array(mask).swapaxes(0, -1) for mask in masks ]	

		return imgs, masks

	@classmethod
	def load_data(cls, imgs_path:list, masks_path:str, size:tuple, method='crop'):
		imgs = []
		masks = []

		if method == 'crop':
			for img_path in imgs_path:
				x, y = cls.load_img_and_masks_crop(img_path, masks_path, size)
				imgs.extend(x)
				masks.extend(y)
			
		elif method == 'resize':
			for img_path in imgs_path:
				x, y = cls.load_img_and_masks_resize(img_path, masks_path, size)
				imgs.append(x)
				masks.append(y)

		return imgs, masks