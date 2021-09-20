一、把json文件批量转化为png图片的方法：
   1、找到\image_dataset_label \json文件夹下的  json2png.sh脚本，这是一个批量转化的脚本程序，要把它放在json文件同级目录下；
   2、打开anaconda, 要事先创建好一个anaconda环境，比如命名为labelme的一个环境，然后在这个环境里安装好labelme，然后执行 
	acrivate labelme 激活labelme环境,然后进入到json2png.sh目录下，然后执行
	json2png.sh 命令即可，

     可参考：https://blog.csdn.net/blue_skyLZW/article/details/120210576?spm=1001.2014.3001.5501
