import glob
import os

files = glob.glob("Images-02_/*.jpg")
files.sort()
for index,file_ in enumerate(files,start=1):
	print(file_)
	id_ = str(120 + index).zfill(3)
	os.rename(file_, 'Images-02_/imagem-{}.jpg'.format(id_))