import xml.etree.ElementTree as ET
name = 'imagem-'
for i in range(0,120):
    filename = name+str('{0:03}'.format(i+1))+".xml"
    print(filename)
    xmlTree = ET.parse(filename)
    rootElement = xmlTree.getroot()
    for element in rootElement.findall("object/polygon"):
        #Find the book that has title as 'Python Tutorial'
        element.find('username').text = "anonymous"
    #Write the modified xml file.        
    xmlTree.write(filename,encoding='UTF-8',xml_declaration=True)
    