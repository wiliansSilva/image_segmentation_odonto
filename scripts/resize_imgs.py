from skimage import io, util, transform
import os, sys, glob
from tqdm import tqdm
import threading

CLASSES = ['root canal treatment', 'restoration', 'crown', 'dental implant', 'tooth', 'pulp']

def resize_img(img_filename, masks_path, save_imgs_path, save_masks_path, size):
    image_id = img_filename.split('/')[-1].split('.')[0]

    img = io.imread(img_filename, as_gray=True)
    masks = [ None for _ in CLASSES]
            
    masks_filenames = glob.glob(os.path.join(masks_path, f"{image_id}", "*"))

    resized_img = transform.resize(img, (size, size))

    for mask_filename in masks_filenames:
        class_ = mask_filename.split("/")[-1].split(".")[0]
        mask = io.imread(mask_filename, as_gray=True)
        masks[CLASSES.index(class_)] = transform.resize(mask, (size, size))

    try:
        io.imsave(os.path.join(save_imgs_path, f"{image_id}.jpg"), util.img_as_ubyte(resized_img), check_contrast=False)
    except FileExistsError:
        print(f"O arquivo {os.path.join(save_imgs_path, f'{image_id}.jpg')} já existe!")
    
    for i, mask in enumerate(masks):
        if not os.path.exists(os.path.join(save_masks_path, f"{image_id}")):
            os.makedirs(os.path.join(save_masks_path, f"{image_id}"))
        
        try:
            io.imsave(os.path.join(save_masks_path, f"{image_id}", f"{CLASSES[i]}.png"), util.img_as_ubyte(mask), check_contrast=False)
        except FileExistsError:
            print(f"O arquivo {os.path.join(save_masks_path, f'{image_id}', f'{CLASSES[i]}.png')} já existe!")

def resize_imgs(images_filenames, masks_path, save_imgs_path, save_masks_path, size):	
    for img_filename in images_filenames:
        resize_img(img_filename, masks_path, save_imgs_path, save_masks_path, size)

    threads = []
    for img_filename in images_filenames:
        thread = threading.Thread(target=resize_img, args=(img_filename, masks_path, save_imgs_path, save_masks_path, size))
        threads.append(thread)

    for thread in threads:
        thread.join()

if __name__ == '__main__':
    images_filenames = glob.glob(os.path.join(sys.argv[1], "*"))

    if not os.path.exists(sys.argv[3]):
        os.makedirs(sys.argv[3])

    if not os.path.exists(sys.argv[4]):
        os.makedirs(sys.argv[4])

    threads = []
    for i in range(8):
        thread = threading.Thread(target=resize_imgs, args=(images_filenames[i*len(images_filenames):(i+1)*len(images_filenames):], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()


