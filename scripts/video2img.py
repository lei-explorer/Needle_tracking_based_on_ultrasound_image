#!/usr/bin/env python
# coding: utf-8

import cv2 as cv
import os

video_list = []
for i in range(1,14):
    video_list.append(str(i)+'.mp4')
absolute_path = os.path.dirname(__file__) #获取本文件的绝对路径
absolute_path = absolute_path[:-len('/scripts')]
print(absolute_path)
video_num = 12 #视频序号,0-12开始
videoCapture = cv.VideoCapture(absolute_path +'/video/'+video_list[video_num])
i ,num = 0,0
faile = 0
def save_image(image , path , num):
    path = path + str(video_num)+'_'+str(num)+ '.jpg'
    cv.imwrite(path , image)

def img_crop(frame):
    frame_crop = frame[200:600,:,:] #row
    frame_crop = frame_crop[:,540:1120,:] #column
    return frame_crop

if __name__ == '__main__':
    while True:
        ret, frame = videoCapture.read() #frame是每一帧的图像，是个三维矩阵
        num +=1
        if ret and num%4 == 0: #每4张存一次
            frame = img_crop(frame)
            save_image(frame,'./image_dataset/',i)
            print('save image - ',i)
            i += 1
        elif not ret:
            break

