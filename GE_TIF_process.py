import time, random, sys, os, math
from PIL import Image
from scipy import misc
from PIL import Image
import pandas as pd
import numpy as np
from PySide6.QtWidgets import QFileDialog
import tkinter as tk
from tkinter import filedialog
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import cv2
from tifffile import TiffFile,tifffile

# for tinker interface
root = tk.Tk()
root.withdraw()

def open_tif_img(img_path:str=None):
    """
    open a 16bit tif file and return filename,img_data
    :return: filename,img_data in np array
    """
    img_data = np.array([])
    if  not img_path or not os.path.isdir(img_path):
        img_path='/'
    filename = filedialog.askopenfilename(initialdir = img_path,title = "Select tif file",filetypes = (("tif file","*.tif"),("jpeg files","*.jpg")))
    print(f'get file: {filename}')
    foldername, filetype=os.path.splitext(filename)
    print(foldername, filetype)
    if os.path.isfile(filename) and filename.endswith('.tif'):
        img = Image.open(filename)
        img_data = np.array(img,dtype=np.float32)
        print(f'shape of the read img={np.shape(img_data)}')
    return filename, img_data

def cv2_filter_img(img_data=np.array([])):
    #img=cv2.cvtColor(img_data,cv2.CV_32F)
    # 均值滤波
    # 用3*3的核对图片进行卷积操作，核上的参数都是1/9，达到均值的效果
    blur = cv2.blur(img_data, (3, 3))
    # 方框滤波（归一化）=均值滤波
    box1 = cv2.boxFilter(img_data, -1, (3, 3), normalize=True)
    # 方框滤波（不归一化）
    #box2 = cv2.boxFilter(img_data, -1, (3, 3), normalize=False)
    # 高斯滤波
    # 用5*5的核进行卷积操作，但核上离中心像素近的参数大。
    guassian = cv2.GaussianBlur(img_data, (5, 5), 1)
    # 中值滤波
    # 将某像素点周围5*5的像素点提取出来，排序，取中值写入此像素点。
    mean3 = cv2.medianBlur(img_data, 3)
    mean5 = cv2.medianBlur(img_data, 5)

    # 展示效果
    titles = ['Original figure', 'blur', 'box_norm', 'guassian','mean3','mean5']
    images = [img_data, blur, box1, guassian,mean3,mean5]
    for i in range(6):
        #plt.subplot(3, 2, i+1), plt.imshow(images[i],cmap=cm.rainbow,vmax=1400,vmin=1200)
        plt.figure(f'{titles[i]}'), plt.imshow(images[i],cmap=cm.rainbow,vmax=1400,vmin=1250)
        #plt.title(titles[i])
        #plt.xticks([]), plt.yticks([])
    plt.show()

def cv2_mean_filter(img_file:str,filter_N:int=3):
    """_summary_
    filter the input tif image by cv2.medianBlur and save 
    Args:
        img_file (str): tif_img file
        filter_N (int): cv2 medianblur parameter 3,5,7... Defaults to 3.
    """
    if os.path.isfile(img_file) and img_file.endswith('.tif'):
        img = Image.open(img_file)
        img_data = np.array(img,dtype=np.float32)
        print(f'shape of the read img={np.shape(img_data)}')
        mean_img=cv2.medianBlur(img_data, filter_N)
        titles=["origin","Medianblur"]
        plt.subplot(1, 2, 1),plt.imshow(img_data,cmap=cm.rainbow,vmin=1300,vmax=1400)
        plt.title("origin")
        plt.subplot(1, 2, 2),plt.imshow(mean_img,cmap=cm.rainbow,vmin=1300,vmax=1400)
        plt.title("Medianblur")
        plt.show()
        # save to tif image
        filename = filedialog.asksaveasfilename(title=u'保存tif图片', filetypes=[("tiff", ".tif")])
        save_tif_data(f'{filename}.tif',np.array(mean_img,dtype=np.float32))

def save_tif_data(filename:str,img_data:np.array([])):
        """
        save data in the main figure box
        :return:
        """
        if filename.endswith('.tif'):
            tifffile.imsave(filename, img_data)




if __name__ == "__main__":
    #download_path = r'F:\BeautifulPictures'
    download_path=r'E:\迅雷下载\MaryMoody'
    save_path = r'F:\Beautyleg'
    img_path=r'E:\areaDetector\saved data'
    tif_path=r'E:/areaDetector/saved data/20220822/GR-X-7670-600s_line .tif'
    tif_file, img_data=open_tif_img(img_path)
    pd_img=pd.DataFrame(img_data)
    #print(f'get tif pic:{tif_file} with image date {pd_img}')
    img_series=img_data.flatten()
    pd_img_series=pd.DataFrame(img_series)
    #cv2_filter_img(img_data)
    cv2_mean_filter(tif_file)
    # plt.imshow(img_data,cmap=cm.rainbow,vmax=1400,vmin=1200)
    # #pd_img_series.plot.kde()
    # print(img_series)
    # plt.show()
    
