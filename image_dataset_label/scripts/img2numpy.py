#! usr/bin/env python
# coding:utf-8

import cv2 as cv
import numpy
import os

this_file_path=os.path.dirname(__file__)
img_ypath=this_file_path[:-len('scripts')]+'/img_y'
img_y_allfile=os.listdir(img_ypath)

img = cv.imread(os.path.join(img_ypath,img_y_allfile[0]),flags=2) #flags=1以原图像读 2以灰度图读入 参考：https://blog.csdn.net/qq_42079689/article/details/102535329

cv.imshow('image',img)
cv.waitKey(0)