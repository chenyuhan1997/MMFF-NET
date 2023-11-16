import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import glob
import random

random.seed(1143)
#-----------------------------------------------------------------------------------------------------------------------
def populate_train_list(lowlight_images_path):
	image_list_lowlight = glob.glob(lowlight_images_path + "*.jpg")
	train_list = image_list_lowlight
	random.shuffle(train_list)
	return train_list
#-----------------------------------------------------------------------------------------------------------------------
class lowlight_loader(data.Dataset):
	def __init__(self, lowlight_images_path, resize = None):
		self.train_list = populate_train_list(lowlight_images_path)
		self.data_list = self.train_list
		self.resize = resize
		print("Total training examples:", len(self.train_list))
	def __getitem__(self, index):
		tf = transforms.Compose([
			transforms.Grayscale(num_output_channels=1),  # 彩色图像转灰度图像num_output_channels默认1
			transforms.Resize(self.resize),
			transforms.ToTensor()
		])
		data_lowlight_path = self.data_list[index]
		data_lowlight = Image.open(data_lowlight_path)
		data_lowlight = tf(data_lowlight)
		return data_lowlight
	def __len__(self):
		return len(self.data_list)