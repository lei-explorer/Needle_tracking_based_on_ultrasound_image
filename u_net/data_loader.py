import os
import random
from random import shuffle
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image # Python Image Library

class ImageFolder(data.Dataset):
	def __init__(self, root,image_size=224,mode='train',augmentation_prob=0.5):
		"""Initializes image paths and preprocessing module."""
		self.root = root  #root = image_path = './dataset/train/'
		
		# GT : Ground Truth 在监督学习中，GT表示对样本的正确标注 即y
		# self.GT_paths = root[:-1]+'_GT/' #root[:-1]去掉路径最后的“ / ” ，在root同级路径下新创建一个xxx_GT/的文件夹
		self.GT_paths = []
		# 最后返回的是root文件夹下所有文件的相对路径组成的列表，路径最后带有文件类型 例如1.jpg
		self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))  #lambda 来创建匿名函数 ,map()把后一个映射到前一个方法中
		self.x_image_paths = []
		for i in self.image_paths:
			if 'matte' in i:
				self.GT_paths.append(i)
			else:
				self.x_image_paths.append(i)
		# print(self.x_image_paths)
		self.image_size = image_size #defult 224 后续没有出现，应该没什么用
		self.mode = mode
		self.RotationDegree = [0,90,180,270]
		self.augmentation_prob = augmentation_prob
		print("image count in {} path :{}".format(self.mode,len(self.image_paths)/2))

	def __getitem__(self, index):
		"""Reads an image from a file and preprocesses it and returns."""
		image_path = self.x_image_paths[index] #逐个图片的相对路径 root文件夹下
		# filename = image_path.split('_')[-1][:-len(".jpg")] #这里取出每张图片的名字（不要.jpg）
		# GT_path = self.GT_paths + 'ISIC_' + filename + '_segmentation.png' #xxx_GT/ISIC_imgname_segmentation.png
		GT_path = self.GT_paths[index]
		image = Image.open(image_path)
		GT = Image.open(GT_path)

		aspect_ratio_original = image.size[1]/image.size[0] #img的纵横比,size属性返回的是w,h

		Transform = [] #存储数据增强不同的变换操作
		p_transform = random.random() #在[0,1)范围内随机生成一个实数
		if (self.mode == 'train') and p_transform <= self.augmentation_prob:
			image = F.hflip(image) #水平翻转图片
			GT = F.hflip(GT)
			# Transform = T.ColorJitter(brightness=0.2,contrast=0.2,hue=0.02) #改变图像属性：亮度（brightness）、对比度（contrast）、饱和度（saturation）和色调（hue)
			# image = Transform(image)

		Transform.append(T.Resize((int(256*aspect_ratio_original)-int(256*aspect_ratio_original)%16,256))) #改变图片尺寸
		Transform.append(T.ToTensor()) #将图片转换成形状为(C,H, W)的Tensor格式 把灰度范围从0-255变换到0-1之间 https://www.pianshen.com/article/6972192583/
		Transform = T.Compose(Transform) #torchvision.transforms.Compose()类的主要作用是串联多个图片变换的操作
		
		image = Transform(image)
		GT = Transform(GT)
		Norm_ = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #把0-1变换到(-1,1)
		image = Norm_(image)
		# print('image=',image)
		return image, GT

	def __len__(self):
		"""Returns the total number of font files."""
		return len(self.x_image_paths)  #数据集的样本数量

def get_loader(image_path, image_size, batch_size, num_workers=2, mode='train',augmentation_prob=0.5):
	"""Builds and returns Dataloader."""
	
	dataset = ImageFolder(root = image_path, image_size =image_size, mode=mode,augmentation_prob=augmentation_prob)
	data_loader = data.DataLoader(dataset=dataset,
								  batch_size=batch_size,
								  shuffle=True, #用于打乱数据集，每次都会以不同的顺序返回(先打乱顺序 再返回batch)
								  num_workers=num_workers)
	return data_loader

if __name__ == '__main__':
	train_loader = get_loader(image_path='./dataset/train/',
                            image_size=224,
                            batch_size=8,
                            num_workers=1,
                            mode='train',
                            augmentation_prob=0.5)
	for i, (images, GT) in enumerate(train_loader):
		#images是tensor 共四个元素：batch_size、channel、h、w
		print(type(images))
		print('i , data_loader.size = ',i,images.shape)
