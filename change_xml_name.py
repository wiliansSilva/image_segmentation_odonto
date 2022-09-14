import glob
import os

files = glob.glob("Annotations-02--03-05-2022/*.xml")
for index,file_ in enumerate(files,start=1):
	print(file_)
	id_ = file_.split('/')[-1].replace('.xml', '').zfill(3)
	os.rename(file_, 'Annotations-02--03-05-2022/imagem-{}.xml'.format(id_))