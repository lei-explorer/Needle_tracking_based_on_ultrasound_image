#!/usr/bin/env python
# coding : utf-8

import os
import shutil

absolute_path = os.path.dirname(__file__) #获取本文件的绝对路径
absolute_path = absolute_path[:-len('/scripts')]
all_file_name = os.listdir(os.path.join(absolute_path,'image_dataset'))
json_file = [] #存储json文件名字
for i in all_file_name:
    if i[-4:] == 'json':
        json_file.append(i)
print(json_file)
for i in json_file:
    shutil.copy(os.path.join(absolute_path+'/image_dataset',i),os.path.join(absolute_path,'image_dataset_label')) #把文件复制到文件夹下