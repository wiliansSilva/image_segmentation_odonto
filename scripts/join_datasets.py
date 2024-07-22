import glob
import os
import argparse

def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset1', '-d1')
	parser.add_argument('--dataset2','-d2')
	parser.add_argument('--new_location', '-nl')
	return parser.parse_args()

if __name__ == "__main__":
	args = parse_arguments()

	dataset1_filenames = glob.glob(os.path.join(args.dataset1, '*'))
	print(len(dataset1_filenames))

	if not os.path.exists(args.new_location):
		os.makedirs(args.new_location)

	if not os.path.exists('./tmp/'):
		os.mkdir('./tmp/')

	os.system(f"cp -r {os.path.join(args.dataset1, '*')} {args.new_location}")

	dataset2_filenames = glob.glob(os.path.join(args.dataset2, '*'))
	print(dataset2_filenames)

	for i, filename in enumerate(dataset2_filenames):
		#extension = filename.split('/')[-1].split('.')[-1]
		#print(str(i+first_file_id).zfill(3))
		os.system(f"cp -r {filename} ./tmp/imagem-{str(i+121).zfill(3)}")
		print(i+120)
		#os.renames(filename, f"./tmp/imagem-{str(i+first_file_id).zfill(3)}.{extension}")

	os.system(f"cp -r ./tmp/* {args.new_location}")
