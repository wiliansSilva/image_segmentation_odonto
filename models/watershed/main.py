import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, measure, segmentation, morphology

# Thanks to https://github.com/bnsreenu/python_for_microscopists/blob/master/205_predict_unet_with_watershed_single_image.py
def instance_segmentation_watershed(img: np.array, clear_border: bool=False, sure_fg_dist=0.2):
    # Threshold image to binary using OTSU. ALl thresholded pixels will be set to 255
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3,3), np.uint8)
    #opening = morphology.opening(thresh, kernel)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)

    if clear_boarder:
        opening = segmentation.clear_border(opening) #Remove edge touching grains.

    sure_bg = cv2.dilate(opening, kernel, iterations=10)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, sure_fg_dist * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 10
    markers[unknown == 255] = 0

    markers = cv2.watershed(img, markers)
    markers[markers == -1] = 10
    markers = markers - 10

    return markers

if __name__ == "__main__":
    pass