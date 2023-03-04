import gdown
import zipfile
import shutil
import os


def extract_data():

	url = "https://drive.google.com/u/0/uc?id=1-w9xsx2FbwFf96A1y1GFcZ3odzdEBves&export=download"
	output = "download_file.zip"

	gdown.download(url, output, quiet=False)

	with zipfile.ZipFile(output, 'r') as zip_ref:
	    zip_ref.extractall('./extracted_data')

	test_zip = "./extracted_data/ShanghaiTech_features/SH_Test_ten_crop_i3d.zip"
	train_zip = "./extracted_data/ShanghaiTech_features/SH_Train_ten_crop_i3d.zip"
	with zipfile.ZipFile(test_zip, 'r') as zip_ref:
	    zip_ref.extractall('./extracted_data')
	with zipfile.ZipFile(train_zip, 'r') as zip_ref:
	    zip_ref.extractall('./extracted_data')


	test_folder = "./extracted_data/SH_Test_ten_crop_i3d/"
	train_folder = "./extracted_data/SH_Train_ten_crop_i3d/"

	test_dest = "./data/shanghaitech/i3d/test/"
	train_dest = "./data/shanghaitech/i3d/train/"

	files = os.listdir(test_folder)
	for fname in files:

		shutil.copy2(os.path.join(test_folder,fname), test_dest)

	files = os.listdir(train_folder)
	for fname in files:

		shutil.copy2(os.path.join(train_folder,fname), train_dest)


if __name__ == '__main__':
	extract_data()

























