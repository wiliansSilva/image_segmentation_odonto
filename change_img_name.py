import glob
import os

files = glob.glob("Images-02/*.jpg")
for index,file_ in enumerate(files,start=1):
	print(file_)
	id_ = file_.split('/')[-1].replace('.jpg', '').replace('imagem-', '').zfill(3)
	os.rename(file_, 'Images-02/imagem-{}.jpg'.format(id_))