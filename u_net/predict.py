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
from torchvision import transforms as T
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm #显示进度条


#对输入图片进行处理
def img_deal(image_path):
    image = Image.open(image_path)
    Transform = []
    aspect_ratio_original = image.size[1] / image.size[0]  # img的纵横比,size属性返回的是w,h
    Transform.append(T.Resize((int(256 * aspect_ratio_original) - int(256 * aspect_ratio_original) % 16, 256)))  # 改变图片尺寸
    Transform.append(T.ToTensor())  # 将图片转换成形状为(C,H, W)的Tensor格式 把灰度范围从0-255变换到0-1之间 https://www.pianshen.com/article/6972192583/
    Transform = T.Compose(Transform)  # torchvision.transforms.Compose()类的主要作用是串联多个图片变换的操作

    image = Transform(image)
    Norm_ = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 把0-1变换到(-1,1)
    image = Norm_(image)
    return image

def build_model(model_type,t=3):
    """Build generator and discriminator."""
    if model_type == 'U_Net':
        unet = U_Net(img_ch=3, output_ch=1)
    elif model_type == 'R2U_Net':
        unet = R2U_Net(img_ch=3, output_ch=1, t=t)
    elif model_type == 'AttU_Net':
        unet = AttU_Net(img_ch=3, output_ch=1)
    elif model_type == 'R2AttU_Net':
        unet = R2AttU_Net(img_ch=3, output_ch=1, t=t)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    unet.to(device)
    return unet


def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()  # image.size() = torch.Size([4, 1, 336, 256])
    image = image[0, :, :, :]
    image = image.squeeze()
    unloader = transforms.Compose([transforms.ToPILImage()])
    image = unloader(image)
    return image


def main(u_net):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    absolute_path = os.path.dirname(__file__)
    img_file = os.listdir(absolute_path+'/dataset/test')
    imgs = list(filter(lambda x:'matte' not in x ,img_file))
    for i in imgs:
        img_path = absolute_path+'/dataset/test/'+i
        # print(img_path)
        image = img_deal(img_path)
        image = Variable(torch.unsqueeze(image, dim=0).float(), requires_grad=False)
        image = image.to(device)
        SR = u_net.forward(image)
        # 保存PIL格式的图片
        if not os.path.exists('./predict_img/'):
            os.makedirs('./predict_img/')
        img_PIL = tensor_to_PIL(SR)
        # print('img_PIL=',np.matrix(img_PIL.getdata())) #PIL的data是1行 192x256列的二维矩阵，数值0-255
        plt.imshow(img_PIL)
        img = plt.gcf()
        img.savefig('./predict_img/'+i[:-4]+'.png')
        img.clear()


if __name__ == "__main__":
    absolute_path = os.path.dirname(__file__)
    u_net_path = absolute_path+'/models'
    u_net_name = os.listdir(u_net_path)
    u_net_path = u_net_path + '/'+u_net_name[0]
    u_net = build_model('U_Net')
    u_net.load_state_dict(torch.load(u_net_path))  # 导入训练好的模型

    main(u_net)