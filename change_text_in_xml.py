import os
import xml.etree.cElementTree as ET
import cv2

dir = 'Annotations-02--03-05-2022/'

for file in os.listdir(dir):
    tree = ET.parse(os.path.join(dir, file))
    root_xml = tree.getroot()

    for folder in root_xml.findall('filename'):
        folder.text = "{}.jpg".format(file[0:10])
    tree.write(os.path.join(dir, file))