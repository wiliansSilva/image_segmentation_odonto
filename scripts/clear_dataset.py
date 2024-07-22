from skimage import io
import numpy as np
import os
import glob

IMAGES_PATH = "../Images/images_splited2"
MASKS_PATH = "../Masks/masks_splited2"

def main():
    masks_folder = glob.glob(os.path.join(MASKS_PATH, "*"))
    #print(masks_folder)
    for mask_folder in masks_folder:
        masks = glob.glob(os.path.join(mask_folder, "*"))
        print(mask_folder)
        delete = True
        for mask in masks:
            file_ = io.imread(mask, as_gray=True)
            result = np.count_nonzero(file_)
            print(result)
            if result != 0:
                delete = False
        if delete:
            id_ = mask_folder.split("/")[-1]
            os.system(f"rm -rf {mask_folder}")
            print(mask_folder)
            os.system(f"rm -rf {os.path.join(IMAGES_PATH, id_)}.jpg")
            print(os.path.join(IMAGES_PATH, id_))

if __name__ == "__main__":
    main()

    #extrapolation