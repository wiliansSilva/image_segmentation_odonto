import xml.etree.ElementTree as ET
import numpy as np
import cv2
import glob
import os


files = glob.glob("Annotations-02--03-05-2022/*.xml")
for index,file in enumerate(files,start=1):
    print(file)
    os.rename(file, 'Annotations-02--03-05-2022/imagem-{0:003}.xml'.format(index))