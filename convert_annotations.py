import xml.etree.ElementTree as ET
import numpy as np
import cv2
import glob

'''
map colors

tooth = white
restoration = blue
crown = yellow
dental implant = green
root canal treatment = purple
pulp = red - IGNORAR 
'''
def main():
    # get all paths to xmls in path
    files = glob.glob("Annotations-02--03-05-2022/*.xml")
    points = []
    for file in files:
        tree = ET.parse(file)
        root = tree.getroot()
        # read image to get shape
        image = cv2.imread(file.replace("Annotations-02--03-05-2022/","Images/").replace("xml","jpg"))
        # create mask image with zeros
        try:
            mask = np.zeros((image.shape))
        # loop to all objects in xml file
        
            for neighbor in root.findall("object"):
                polygon = neighbor.find('polygon')
                # get points
                for pt in polygon.findall('pt'):
                    x = int(round(float(pt.find("x").text)))
                    y = int(round(float(pt.find("y").text)))
                    points.append([x,y])

                # Verify which one polygon is to set color
                if neighbor.find('name').text == "restoration":
                    color = (255,0,0)
                    mask = cv2.fillPoly(mask,np.int32([points]),color)
                if neighbor.find('name').text == "tooth":
                    color = (255,255,255)
                    mask = cv2.drawContours(mask, np.int32([points]), -1, color, 3)
                if neighbor.find('name').text == "crown":
                    color = (0,255,255)
                    mask = cv2.fillPoly(mask,np.int32([points]),color)
                if neighbor.find('name').text == "dental implant":
                    color = (0,128,0)
                    mask = cv2.fillPoly(mask,np.int32([points]),color)
                if neighbor.find('name').text == "root canal treatment":
                    color = (128,0,128)
                    mask = cv2.fillPoly(mask,np.int32([points]),color)
                if neighbor.find('name').text == "pulp":
                    color = (0,0,255)
                    mask = cv2.fillPoly(mask,np.int32([points]),color)
                
                
                points = []
            str_index = file[34:]
            cv2.imwrite("Mask2/imagem-{}.png".format(str_index).replace(".xml",""),mask)
            print("Create Mask from file: ",file)
            mask[:] = 0
        except AttributeError:
            print("shape not found")

if __name__ == '__main__':
    main()
