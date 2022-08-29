import numpy as np
import matplotlib.pyplot as plt
import tkinter
from tkinter import *
from tkinter.filedialog import askopenfilename
import matplotlib.image as mpimage
from PIL import Image
# coding: utf-8
import csv
from scipy import interpolate


info = {}
info['element'] = 'C'
info['energy'] = 445

def fastinterp1(x, y, xi):
    ixi = np.digitize(xi, x)
    n = len(x)
    ixi[ixi == n] = n - 1
    t = (xi - x[ixi-1])/(x[ixi] - x[ixi-1])
    yi = (1-t) * y[ixi-1] + t * y[ixi]
    yi = yi.T
    return yi


def detectorclean(exp, noise1, noise2):

    exp = exp - np.mean(exp[:, noise1:noise2])
    exp[exp > (np.max(exp) * thresholdUP)] = 0
    exp[exp < (np.min(exp) * thresholdDOWN)] = 0
    detectorcleanout = exp
    return detectorcleanout


def clear_bg(exp):
    u, v = exp.shape
    temp = np.zeros((u, v))
    out = np.zeros((u, v))

    for i in np.arange(u):
        k = (np.sum(exp[i, 1:10]) - np.sum(exp[i, -10:-1]))/(v)/10
        b = np.sum(exp[i, 1:10])/10 - k*10
        exp_bg = -k * np.arange(v) + b
        temp[i, :] = exp[i, :] - exp_bg
    #print(type(temp))
    return u, v, temp


root = Tk()
root.withdraw()
root.update()
img_path = askopenfilename(title=u'Read CCD image')
#bg_path = askopenfilename(title=u'Read background image')
root.destroy()

# img_path = r'D:\eline\RXES\code\data\20220102\19_38.3\TIO2@424eV_slit_1000_2000s_R1_38.3_r2.tif'
# # bg_path = r'D:\eline\REXS\code\data\20211215\-10degree_01_BGR_BGR.tif'

#img= mpimage.imread(img_path).astype('float64')
# add by limin
img = Image.open(img_path)
matrix = np.array(img,dtype=np.float32)
#background = mpimage.imread(bg_path).astype('float64')
'''
matrix1 = mpimage.imread(img_path).T
matrix1 = matrix1[:,:200]
# show raw CCD image
'''

# Background subtraction
""" ExposureTime = 600 #seconds
background_aqn_time = 600 #seconds
extract_background=True
if extract_background:
    rawImageData = 1. * matrix
    if background.shape == matrix.shape[:2]:
        if background_aqn_time:
            matrix -= np.array(ExposureTime) * background \
                / background_aqn_time """
matrix1 = matrix.T
#matrix1 = matrix

thresholdUP = 0.9
thresholdDOWN = 0.1
matrix1 = detectorclean(matrix1, noise1=50, noise2=100)
m, n, out = clear_bg(matrix1)
matrix = out
#m2 = round(m/2)
index = 1600
print(index)
new_img = np.zeros((m,n))
for j in range(1,10,1):
    k = 0.4 + j*0.1+j**2*(1e-07)
    
    low_lim = round(index*20 -1200)
    high_lim = round(index*20 + 1200)
    #low_lim = 3640
    #high_lim = 14640
    new_X = np.arange(low_lim, high_lim)
    result = np.zeros(len(new_X))
    xinitial = np.arange(n)
    xinterp = np.arange(0, n, 0.05)
    xx = np.linspace(0, len(xinterp), n)
    for ii in range(m):
        temp = matrix[ii, :]
        ntemp = fastinterp1(xinitial, temp, xinterp)
        dd = round(k*ii)
        new_img[ii,:] = fastinterp1(new_X,ntemp[low_lim-dd:high_lim-dd],xx)
        #f=interpolate.interp1d(new_X,ntemp[low_lim-dd:high_lim-dd],kind='slinear')
        #new_img = f(xx)
        result = result + ntemp[low_lim-dd:high_lim-dd]
    
    y_ = fastinterp1(new_X, result, xx)
    plt.plot(xinitial[round(low_lim/20):round(high_lim/20)],\
            y_[round(low_lim/20):round(high_lim/20)])
    plt.pause(0.5)
plt.show()

