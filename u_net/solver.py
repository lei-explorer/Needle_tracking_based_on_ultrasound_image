import os
import numpy as np
import time
import datetime
import torch
import torchvision
from torchvision import transforms
from torch import optim
from torch.autograd import Variable
from PIL import Image
import torch.nn.functional as F
from evaluation import *
from network import U_Net,R2U_Net,AttU_Net,R2AttU_Net
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm #显示进度条


class Solver(object):
	def __init__(self, config, train_loader, valid_loader, test_loader):
		# Data loader
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.test_loader = test_loader

		# Models
		self.unet = None
		self.optimizer = None
		self.img_ch = config.img_ch
		self.output_ch = config.output_ch
		self.criterion = torch.nn.BCELoss() #返回loss的均值，即loss.mean()
		self.augmentation_prob = config.augmentation_prob

		# Hyper-parameters
		self.lr = config.lr
		self.beta1 = config.beta1
		self.beta2 = config.beta2

		# Training settings
		self.num_epochs = config.num_epochs
		self.num_epochs_decay = config.num_epochs_decay
		self.batch_size = config.batch_size

		# Step size
		self.log_step = config.log_step
		self.val_step = config.val_step

		# Path
		self.model_path = config.model_path
		self.result_path = config.result_path
		self.mode = config.mode

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model_type = config.model_type
		self.t = config.t
		self.build_model()

	def build_model(self):
		"""Build generator and discriminator."""
		if self.model_type =='U_Net':
			self.unet = U_Net(img_ch=3,output_ch=1)
		elif self.model_type =='R2U_Net':
			self.unet = R2U_Net(img_ch=3,output_ch=1,t=self.t)
		elif self.model_type =='AttU_Net':
			self.unet = AttU_Net(img_ch=3,output_ch=1)
		elif self.model_type == 'R2AttU_Net':
			self.unet = R2AttU_Net(img_ch=3,output_ch=1,t=self.t)
			

		self.optimizer = optim.Adam(list(self.unet.parameters()),
									  self.lr, (self.beta1, self.beta2)) #参数含义：https://blog.csdn.net/kgzhang/article/details/77479737
		self.unet.to(self.device)

		# self.print_network(self.unet, self.model_type)

	def print_network(self, model, name):
		"""Print out the network information."""
		num_params = 0
		for p in model.parameters():
			num_params += p.numel()
		print(model)
		print(name)
		print("The number of parameters: {}".format(num_params))

	def to_data(self, x):
		"""Convert variable to tensor."""
		if torch.cuda.is_available():
			x = x.cpu()
		return x.data

	# def update_lr(self, g_lr, d_lr):
	# 	for param_group in self.optimizer.param_groups:
	# 		param_group['lr'] = lr

	def reset_grad(self):
		"""Zero the gradient buffers."""
		self.unet.zero_grad()

	def compute_accuracy(self,SR,GT):
		SR_flat = SR.view(-1)
		GT_flat = GT.view(-1)

		acc = GT_flat.data.cpu()==(SR_flat.data.cpu()>0.5)

	def tensor2img(self,x):
		img = (x[:,0,:,:]>x[:,1,:,:]).float()
		img = img*255
		return img

	def tensor_to_PIL(self,tensor):
		image = tensor.cpu().clone()  # image.size() = torch.Size([4, 1, 336, 256])
		image = image[0,:,:,:]
		image = image.squeeze()
		unloader = transforms.Compose([transforms.ToPILImage()])
		image = unloader(image)
		return image

	def tensor_to_np(self,img_tensor):
		img = img_tensor.mul(255).byte()
		img = img.cpu().numpy().squeeze() #transpose（） 参考：https://blog.csdn.net/Arthur_Holmes/article/details/103307093/
		# image.shape= (4,336, 256) hxwxbatch_size
		return img

	def save_img_arr(self,path,tensor):
		arr = self.tensor_to_np(tensor)
		arr = arr[0,:,:] #只保存batch里的第一张图片
		arr = arr.squeeze()
		print('arr.shape=',arr.shape)
		filename = path + '.txt'
		im = Image.fromarray(arr)
		im = im.convert('L')  # 这样才能转为灰度图，如果是彩色图则改L为‘RGB’
		im.save(path+'.png')
		row = arr.shape[0]
		with open(filename, 'w') as f:  # 若filename不存在会自动创建，写之前会清空文件
			for i in range(0, row):
				for j in arr[i,:]:
					f.write(str(j))
					f.write(' ')
				f.write("\n")
			f.close()


	def train(self):
		"""Train encoder, generator and discriminator."""

		#====================================== Training ===========================================#
		#  .pkl跟.pth 差不多，都是保存神经网络的一种文件格式
		unet_path = os.path.join(self.model_path, '%s-%d-%.4f-%d-%.4f.pkl' %(self.model_type,self.num_epochs,self.lr,self.num_epochs_decay,self.augmentation_prob))

		# U-Net Train
		if os.path.isfile(unet_path): #用于判断某一对象(需提供绝对路径)是否为文件
			# Load the pretrained Encoder
			self.unet.load_state_dict(torch.load(unet_path)) #r如果有预训练好的模型，则load进来
			print('%s is Successfully Loaded from %s'%(self.model_type,unet_path))
		else:
			# Train for Encoder
			lr = self.lr
			best_unet_score = 0.
			each_loss = []
			for epoch in range(self.num_epochs):

				self.unet.train(True) # train(mode=True) 将当前网络模块设置成训练模式
				epoch_loss = 0
				single_loss = 0
				
				acc = 0.	# Accuracy
				SE = 0.		# Sensitivity (Recall) 敏感性
				SP = 0.		# Specificity 特异性
				PC = 0. 	# Precision 精度
				F1 = 0.		# F1 Score
				JS = 0.		# Jaccard Similarity
				DC = 0.		# Dice Coefficient
				length = 0

				for i, (images, GT) in enumerate(tqdm(self.train_loader)):

					# GT : Ground Truth
					images = images.to(self.device)
					GT = GT.to(self.device)

					# SR : Segmentation Result
					SR = self.unet(images)

					SR_probs = F.sigmoid(SR)
					SR_flat = SR_probs.view(SR_probs.size(0),-1)

					GT_flat = GT.view(GT.size(0),-1) #.view()相当于tensor里的reshape() 把每张图片展成一行
					loss = self.criterion(SR_flat,GT_flat) #单张图片像素的平均损失
					epoch_loss += loss.cpu().item()

					# Backprop + optimize
					self.reset_grad() #反向传播之前清空上一次的梯度
					loss.backward()
					self.optimizer.step() #更新

					acc += get_accuracy(SR,GT)
					SE += get_sensitivity(SR,GT)
					SP += get_specificity(SR,GT)
					PC += get_precision(SR,GT)
					F1 += get_F1(SR,GT)
					JS += get_JS(SR,GT)
					DC += get_DC(SR,GT)
					length += images.size(0) #data.DataLoader会把每一个batch内的图片放进一个张量，张量的第一维就是batch的图片数量

					# # 保存预测的numpy
					# if not os.path.exists('./train_predict_np/'):
					# 	os.makedirs('./train_predict_np/')
					# self.save_img_arr('./train_predict_np/'+str(epoch)+'_'+str(i),SR)
					# # 保存PIL格式的图片
					# if not os.path.exists('./train_predict_img/'):
					# 	os.makedirs('./train_predict_img/')
					# img_PIL = self.tensor_to_PIL(SR)
					# # print('img_PIL=',np.matrix(img_PIL.getdata())) #PIL的data是1行 192x256列的二维矩阵，数值0-255
					# plt.imshow(img_PIL)
					# img = plt.gcf()
					# img.savefig('./train_predict_img/'+str(epoch)+'_'+str(i)+'.jpg')
					# img.clear()

				acc = acc/length
				SE = SE/length
				SP = SP/length
				PC = PC/length
				F1 = F1/length
				JS = JS/length
				DC = DC/length
				each_loss.append(epoch_loss / length)


				# Print the log info
				print('Epoch [%d/%d], Loss: %.4f, \n[Training] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f' % (
					  epoch+1, self.num_epochs, \
					  epoch_loss,\
					  acc,SE,SP,PC,F1,JS,DC))

				# Decay learning rate
				if (epoch+1) > (self.num_epochs - self.num_epochs_decay):
					lr -= (self.lr / float(self.num_epochs_decay))
					for param_group in self.optimizer.param_groups:
						param_group['lr'] = lr
					print ('Decay learning rate to lr: {}.'.format(lr))
				
				#===================================== Validation 验证 ====================================#
				# 是模型训练过程中单独留出的样本集,用来在模型迭代训练时，用以验证当前模型泛化能力
				self.unet.train(False)
				self.unet.eval()

				acc = 0.	# Accuracy
				SE = 0.		# Sensitivity (Recall)
				SP = 0.		# Specificity
				PC = 0. 	# Precision
				F1 = 0.		# F1 Score
				JS = 0.		# Jaccard Similarity
				DC = 0.		# Dice Coefficient
				length=0
				# with torch.no_grad():
				for i, (images, GT) in enumerate(self.valid_loader):

					images = images.to(self.device)
					GT = GT.to(self.device)
					SR = torch.sigmoid(self.unet(images))
					acc += get_accuracy(SR,GT)
					SE += get_sensitivity(SR,GT)
					SP += get_specificity(SR,GT)
					PC += get_precision(SR,GT)
					F1 += get_F1(SR,GT)
					JS += get_JS(SR,GT)
					DC += get_DC(SR,GT)
						
					length += images.size(0)
					
				acc = acc/length
				SE = SE/length
				SP = SP/length
				PC = PC/length
				F1 = F1/length
				JS = JS/length
				DC = DC/length
				unet_score = JS + DC

				print('[Validation] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f'%(acc,SE,SP,PC,F1,JS,DC))
				
				'''
				torchvision.utils.save_image(images.data.cpu(),
											os.path.join(self.result_path,
														'%s_valid_%d_image.png'%(self.model_type,epoch+1)))
				torchvision.utils.save_image(SR.data.cpu(),
											os.path.join(self.result_path,
														'%s_valid_%d_SR.png'%(self.model_type,epoch+1)))
				torchvision.utils.save_image(GT.data.cpu(),
											os.path.join(self.result_path,
														'%s_valid_%d_GT.png'%(self.model_type,epoch+1)))
				'''


				# Save Best U-Net model
				# if unet_score > best_unet_score:
				# best_unet_score = unet_score
				# best_epoch = epoch
			best_unet = self.unet.state_dict() #只保存神经网络的模型参数
			print('Best %s model score : %.4f'%(self.model_type,best_unet_score))
			torch.save(best_unet,unet_path)

			plt.figure(figsize=(6, 3))
			plt.title('Loss-num_train', fontsize=22)
			plt.xlabel('num_train', fontsize=22)
			plt.ylabel('Loss', fontsize=22)
			plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
			plt.plot(range(self.num_epochs), each_loss, 'r-')
			filePath = './' + str(self.num_epochs) +'_loss.png'
			plt.savefig(filePath,dpi=600)
			plt.gcf().clear()

	def test(self):
		#===================================== Test ====================================#
		unet_path = os.path.join(self.model_path, '%s-%d-%.4f-%d-%.4f.pkl' % (
		self.model_type, self.num_epochs, self.lr, self.num_epochs_decay, self.augmentation_prob))

		# U-Net Test
		if not os.path.isfile(unet_path):  # 用于判断某一对象(需提供绝对路径)是否为文件
			return 0

		self.build_model()
		self.unet.load_state_dict(torch.load(unet_path)) #导入训练好的模型

		self.unet.train(False)
		self.unet.eval() #测试时添加model.eval(),保证BN(Batch Normalization）层能够用全部训练数据的均值和方差

		acc = 0.	# Accuracy
		SE = 0.		# Sensitivity (Recall)
		SP = 0.		# Specificity
		PC = 0. 	# Precision
		F1 = 0.		# F1 Score
		JS = 0.		# Jaccard Similarity
		DC = 0.		# Dice Coefficient
		length=0
		for i, (images, GT) in enumerate(self.valid_loader):

			images = images.to(self.device)
			GT = GT.to(self.device)
			SR = F.sigmoid(self.unet(images))
			acc += get_accuracy(SR,GT)
			SE += get_sensitivity(SR,GT)
			SP += get_specificity(SR,GT)
			PC += get_precision(SR,GT)
			F1 += get_F1(SR,GT)
			JS += get_JS(SR,GT)
			DC += get_DC(SR,GT)

			length += images.size(0)

		acc = acc/length
		SE = SE/length
		SP = SP/length
		PC = PC/length
		F1 = F1/length
		JS = JS/length
		DC = DC/length
		unet_score = JS + DC
		print('unet_score=',unet_score)
		f = open(os.path.join(self.result_path,'result.csv'), 'a', encoding='utf-8', newline='')
		wr = csv.writer(f)
		wr.writerow([self.model_type,acc,SE,SP,PC,F1,JS,DC,unet_score,self.lr,self.num_epochs,self.num_epochs_decay,self.augmentation_prob])
		f.close()

