# coding: utf-8
from genericpath import isfile
import numpy as np
import matplotlib.pyplot as plt
import tkinter,time,os
from tkinter import *
from tkinter.filedialog import askopenfilename
import matplotlib.image as mpimage
from PIL import Image
import csv,cv2
from scipy import interpolate
import matplotlib.cm as cm
from scipy.optimize import curve_fit

def fastinterp1(x, y, xi):
    ixi = np.digitize(xi, x)
    n = len(x)
    ixi[ixi == n] = n - 1 
    t = (xi - x[ixi-1])/(x[ixi] - x[ixi-1])
    yi = (1-t) * y[ixi-1] + t * y[ixi]
    yi = yi.T
    return yi

def open_tif():
    """open a tif image save by GE CCD
    return the image path
    """
    root = Tk()
    root.withdraw()
    root.update()
    img_path = askopenfilename(title=u'Read CCD image')
    #bg_path = askopenfilename(title=u'Read background image')
    root.destroy()
    return img_path if os.path.isfile(img_path) and img_path.endswith(".tif") else None

def visualize_tif(img_src:str):
    """view the input image

    Args:
        img_src (str): path to tif image
    """
    if os.path.isfile(img_src) and img_src.endswith(".tif"):
        img = Image.open(img_src)
        raw_matrix = np.array(img,dtype=np.float32)
        fig1 = plt.figure(figsize =(16, 9))
        fig1.canvas.manager.window.setWindowTitle("Visualize raw image")
        plt.subplot(2,2,1),plt.imshow(raw_matrix,cmap=cm.rainbow,vmin=1300,vmax=1400)
        plt.colorbar(location='right', fraction=0.1),plt.title("raw image")
        #matrix1 = matrix.T
        filter_N=3
        #mean_matrix=cv2.medianBlur(raw_matrix, filter_N)
        #mean_matrix = cv2.GaussianBlur(raw_matrix, (5, 5), 1)
        #mean_matrix = cv2.blur(raw_matrix, (3, 3))
        # 方框滤波（归一化）=均值滤波
        mean_matrix = cv2.boxFilter(raw_matrix, -1, (3, 3), normalize=True)
        plt.subplot(2,2,2),plt.imshow(mean_matrix,cmap=cm.rainbow,vmin=1300,vmax=1400)
        plt.colorbar(location='right', fraction=0.1),plt.title("median blur image")
        sum_rows_raw=np.sum(raw_matrix,axis=0)
        sum_rows_mean=np.sum(mean_matrix,axis=0)
        sum_cols_raw=np.sum(raw_matrix,axis=1)
        sum_cols_mean=np.sum(mean_matrix,axis=1)
        
        plt.subplot(2,2,3),plt.title("sum cols")
        plt.plot(sum_rows_raw)
        plt.plot(sum_rows_mean)
        plt.legend(['raw','median blur'])
        
        plt.subplot(2,2,4),plt.title("sum rows")
        plt.plot(sum_cols_raw)
        plt.plot(sum_cols_mean)
        plt.legend(['raw','median blur'])
        plt.show()

def gaussian_fit(x, amplitude, mean, stddev):
    return amplitude * np.exp(-2*((x - mean) / stddev)**2)
#popt, _ = curve_fit(gaussian_fit, x, data)

def fit_gaussian(x,*param):
    return param[0]*np.exp(-np.power(x - param[2], 2.) / (2 * np.power(param[4], 2.)))+\
           param[1]*np.exp(-np.power(x - param[3], 2.) / (2 * np.power(param[5], 2.)))
 #popt,pcov = curve_fit(gaussian,x,y,p0=[3,4,3,6,1,1])

def gaussfit(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

#popt,pcov = curve_fit(gaus,x,y,p0=[1,mean,sigma])


if __name__ == "__main__":
    # num=9
    # x=np.arange(num)
    # # random number
    # y=np.random.randn(num)*10
    # xinterp = np.arange(0, num, 0.05)
    # print(len(y))
    # y_interp=fastinterp1(x,y,xinterp)
    # print(len(y_interp))
    # img_src=open_tif()
    # visualize_tif(img_src)
    # plt.plot(x,y,marker='o', markersize=2)
    # plt.plot(xinterp,y_interp,marker='x', markersize=2)
    # plt.legend(["initial","fast interpolate"])
    # plt.show()
    x0 = np.asarray(range(10))
    y0 = np.asarray([0,1,2,3,4,5,4,3,2,1])
    print(y0)
    n=len(x0)
    mean=sum(x0*y0)/n
    sigma=sum(y0*(x0-mean)**2)/n
    popt,pcov = curve_fit(gaussian_fit,x0,y0,p0=[1,mean,sigma])
    plt.plot(x0,y0,'b+:',label='data')
    plt.plot(x0,gaussian_fit(x0,*popt),'ro:',label='fit')
    plt.legend() 
    plt.title('Fig. 3 - Fit for Time Constant')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.show()