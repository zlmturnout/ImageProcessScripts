# coding: utf-8
import os, sys,time,datetime
import numpy as np
import matplotlib.pyplot as plt
import tkinter
from tkinter import *
from tkinter.filedialog import askopenfilename
import matplotlib.image as mpimage
from PIL import Image
import csv,cv2
from scipy import interpolate
import matplotlib.cm as cm
import pandas as pd
from numpy import exp, loadtxt, pi, sqrt
from lmfit import Model

fig1 = plt.figure(figsize =(16, 9))
fig1.canvas.manager.window.setWindowTitle("Visualize raw image")

# read image
img_path_11=r'I:\\Coding\\CCD\\20220918\\11.tif'
img_11 = Image.open(img_path_11)
matrix_11 = np.array(img_11,dtype=np.float32)
print(matrix_11)
img_path_12=r'I:\\Coding\\CCD\\20220918\\12.tif'
img_12 = Image.open(img_path_12)
matrix_12 = np.array(img_12,dtype=np.float32)
img_path_13=r'I:\\Coding\\CCD\\20220918\\13.tif'
img_13 = Image.open(img_path_13)
matrix_13 = np.array(img_13,dtype=np.float32)

plt.subplot(2,2,1),plt.imshow(matrix_11,cmap=cm.rainbow,vmin=1300,vmax=1350)
plt.colorbar(location='right', fraction=0.1),plt.title("raw image 11")
plt.subplot(2,2,2),plt.imshow(matrix_12,cmap=cm.rainbow,vmin=1300,vmax=1350)
plt.colorbar(location='right', fraction=0.1),plt.title("raw image 12")
plt.subplot(2,2,3),plt.imshow(matrix_13,cmap=cm.rainbow,vmin=1300,vmax=1350)
plt.colorbar(location='right', fraction=0.1),plt.title("raw image 13")
# sum matrix
sum_matrix=(matrix_11+matrix_12+matrix_13)/3

#save sum image
sum_tif=new_tif=os.path.join(os.path.join('./GE/tif_files'),f'sum_11-13_mean.tif')
sum_image=Image.fromarray(sum_matrix)
sum_image.save(new_tif)
plt.subplot(2,2,4),plt.imshow(sum_matrix,cmap=cm.rainbow,vmin=1300,vmax=1350)
plt.colorbar(location='right', fraction=0.1),plt.title("raw sum image")
plt.show()
