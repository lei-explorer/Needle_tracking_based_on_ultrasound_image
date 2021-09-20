#!/usr/bin/env pyhton
# coding:utf-8


import os
import shutil

absolute_path = os.path.dirname(__file__)
absolute_path = absolute_path[:-len('/scripts')]
json_all_file = []
before_all_file = []
for i in os.listdir(absolute_path+'/before'):
    if '_json' in i:
        before_all_file.append(i)

for i in os.listdir(absolute_path+'/json'):
    if '_json' in i:
        json_all_file.append(i)


def movejson_2img_y():
    for i in json_all_file:
        shutil.copy(absolute_path+'/json/'+i+'/label.png',os.path.join(absolute_path, 'img_y'))  # 把文件复制到文件夹下
        os.rename(absolute_path+'/img_y/label.png',absolute_path+'/img_y/'+i[:-5]+'_matte.png')


def movejson_2img_x():
    for i in json_all_file:
        shutil.copy(absolute_path+'/json/'+i+'/img.png',os.path.join(absolute_path, 'img_x'))  # 把文件复制到文件夹下
        os.rename(absolute_path+'/img_x/img.png',absolute_path+'/img_x/'+i[:-5]+'.png')


def movefefore_2test():
    for i in before_all_file:
        shutil.copy(absolute_path+'/before/'+i+'/label.png',os.path.join(absolute_path, 'test'))  # 把文件复制到文件夹下
        os.rename(absolute_path+'/test/label.png',absolute_path+'/test/'+i[:-5]+'_matte.png')
        shutil.copy(absolute_path + '/before/' + i + '/img.png', os.path.join(absolute_path, 'test'))  # 把文件复制到文件夹下
        os.rename(absolute_path + '/test/img.png', absolute_path + '/test/' + i[:-5] + '.png')

def file_rename(filepath):
    file_name=os.listdir(filepath)
    for i in file_name:
        os.rename(os.path.join(filepath,i), filepath+'/'+i[:-4] + '_matte.png')

if __name__ == "__main__":
    # movefefore_2test()
    movejson_2img_y()






