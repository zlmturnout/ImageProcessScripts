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


def detectorclean(exp, noise1, noise2,thresholdUP=0.9,thresholdDOWN=0.1):
    exp = exp - np.mean(exp[:, noise1:noise2])
    exp[exp > (np.max(exp) * thresholdUP)] = 0
    exp[exp < (np.min(exp) * thresholdDOWN)] = 0
    detectorcleanout = exp
    return detectorcleanout

def median_filter(matrix:np.array([]),filter_N:int=3):
    median_matrix=cv2.medianBlur(matrix, filter_N)
    return median_matrix

def clear_bg(exp):
    u, v = exp.shape
    temp = np.zeros((u, v))
    out = np.zeros((u, v))

    for i in np.arange(u):
        k = (np.sum(exp[i, 1:10]) - np.sum(exp[i, -10:-1]))/(v)/10
        b = np.sum(exp[i, 1:10])/10 - k*10
        exp_bg = -k * np.arange(v) +b+exp[i, 0]/10
        temp[i, :] = exp[i, :] - exp_bg
    return u, v, temp

def gaussian(x, amp, cen, wid):
    """1-d gaussian: gaussian(x, amp, cen, wid)"""
    return (amp / (sqrt(2*pi) * wid)) * exp(-(x-cen)**2 / (2*wid**2))

def save_raw_img(raw_matrix:np.array([]),img_file:str):
        """

        Args:
            raw_img_data (np.array): _description_
        """
            # plot the raw image
        fig = plt.figure(figsize =(16, 9))
        fig.canvas.manager.window.setWindowTitle("Visualize raw image")
        plt.subplot(2,2,1),plt.imshow(raw_matrix,cmap=cm.rainbow,vmin=1300,vmax=1400)
        plt.colorbar(location='right', fraction=0.1),plt.title("raw image")
        sum_rows_raw=np.sum(raw_matrix,axis=0)
        row_index=[i for i in range(len(sum_rows_raw)) ]
        sum_cols_raw=np.sum(raw_matrix,axis=1)
        col_index=[j for j in range(len(sum_cols_raw)) ]
        plt.subplot(2,2,3),plt.plot(row_index,sum_rows_raw),plt.title("sum cols")
        plt.subplot(2,2,2),plt.plot(col_index,sum_cols_raw),plt.title("sum rows")
        plt.savefig(img_file)

def Gaussian_FWHM(x,y,center=1200,index=1):
    """find FWHM from the imported pd_data [x,y]

    Args:
        pd_data (_type_): _description_
    """
    
    # x = pd_data.values[:, 0]
    # y = pd_data.values[:, 1]
    gmodel = Model(gaussian)
    result = gmodel.fit(y, x=x, amp=49146, cen=center, wid=2.6)
    print(result.values)
    print(result.fit_report())
    wid_fit=result.params['wid'].value
    wid_err=result.params['wid'].stderr
    cen_fit=result.params['cen'].value 
    cen_err=result.params['wid'].stderr
    print(f'wid_err:{wid_err}')
    if wid_err!= None:
        FWHM=wid_fit*2*np.sqrt(np.log(4))
        FWHM_err=wid_err*2*np.sqrt(np.log(4))
        cen_text=f'x0={cen_fit:4f} +/-{cen_err:.4f}\n'
        wid_text=f'w={wid_fit:4f} +/-{wid_err:.4f}\n'
        FWHW_text=f'FWHM={FWHM:.4f} +/-{FWHM_err:.4f}'
    else:
        FWHM_err='estimate failed'
        cen_text=f'x0={cen_fit:4f} estimate failed\n'
        wid_text=f'w={wid_fit:4f} estimate failed\n'
        FWHW_text=f'FWHM={FWHM:.4f} +/-{FWHM_err}'
    print(f'get FWHM={FWHM:.4f} with error +/-{FWHM_err}')
    fig=plt.figure(figsize =(16, 9))
    fig.canvas.manager.window.setWindowTitle(f"Fit-FWHM-{index}")
    ax=plt.subplot()
    plt.plot(x, y, 'o')
    plt.plot(x, result.init_fit, '--', label='initial fit')
    plt.plot(x, result.best_fit, '-', label='best fit')
    plt.text(0.5, 0.5, s=cen_text+wid_text+FWHW_text,color = "m", transform=ax.transAxes,fontsize=15)
    plt.text(0.5, 1.0, s=f'Gauss FIt-{index}',color = "m", transform=ax.transAxes,fontsize=15)
    plt.legend()
    return FWHM,FWHM_err

def shift_pixel(index:int,j:int=1):
    #return round(0.005 + index*0.01+index**2*(1+j*0.1)*1e-7)
    #return round((0.001+0.005*j)*index+index**2*(1e-7))
    #return round((0.005+0.0005*j)*index+index**2*(1e-7)) # best fit results0918-11
    return round((-0.0007+0.00005*j)*index+index**2*(1e-7)) # best fit results 0918-11
    #return round((-0.0006+0.00005*j)*index+index**2*(1e-7)) # best fit results 0905-12
    #return round(-(20/57.0+0.002*j+j**2*(1e-7))*index)  # for test line y=57*x-57000

def shift_arrray(array:np.array([]),n:int=0):
    """shift a array by n position to left if True, else right

    Args:
        array (np.array): 1D array data
        n (int, optional): how many position Defaults to 0.positive is shift left else right
    """
    return np.append(array[n:],array[:n])


def Fit_peak_data(img_matrix:np.asarray([]),p_col:int,half_n:int,save_folder:str,filename:str):
    """
    img_matrix should contain peak in the middle
    # selected point near the mid of the line
    """
    
    header_list=['pixels']
    cut_matrix = np.array(img_matrix,dtype=np.float32)
    fig1 = plt.figure(figsize =(16, 9)) 
    fig1.canvas.manager.window.setWindowTitle("image data preprocess")
    plt.subplot(3,3,1),plt.imshow(cut_matrix.T,cmap=cm.rainbow,vmin=1300,vmax=1400)
    plt.colorbar(location='bottom', fraction=0.1),plt.title("cut raw image")
    plt.subplot(3,3,4),plt.hist(cut_matrix.flatten(),bins=100),plt.title("intensity histogram")
    # sum columm plot
    sum_rows_cut=np.sum(cut_matrix,axis=0)
    row_index=[i for i in range(len(sum_rows_cut)) ]
    sum_cols_cut=np.sum(cut_matrix,axis=1)
    col_index=[j for j in range(len(sum_cols_cut)) ]
    plt.subplot(3,3,7),plt.plot(col_index,sum_cols_cut),plt.title("sum cols")
    # median filter to denoise
    mean_matrix=median_filter(cut_matrix, filter_N=3)
    # plot the results
    plt.subplot(3,3,2),plt.imshow(mean_matrix.T,cmap=cm.rainbow,vmin=1300,vmax=1400)
    plt.colorbar(location='bottom', fraction=0.1),plt.title("median blur image")
    plt.subplot(3,3,5),plt.hist(mean_matrix.flatten(),bins=200),plt.title("intensity histogram")
    sum_cols_cut=np.sum(mean_matrix.T,axis=1)
    col_index=[j for j in range(len(sum_cols_cut)) ]
    plt.subplot(3,3,8),plt.plot(col_index,sum_cols_cut),plt.title("sum cols")
    # dectect clean--clear noise
    thresholdUP = 0.9
    thresholdDOWN = 0.1
    matrix1 = detectorclean(mean_matrix, noise1=50, noise2=200)
    print(type(matrix1),matrix1.shape)
    row, column, cor_matrix = clear_bg(matrix1) #2052*400
    print(f'row: {row}\ncolumn: {column}')
    plt.subplot(3,3,3),plt.imshow(cor_matrix.T,cmap=cm.rainbow,vmin=-25,vmax=25),plt.title("clear background")
    plt.colorbar(location='bottom', fraction=0.1)
    plt.subplot(3,3,6),plt.hist(cor_matrix.flatten(),bins=200),plt.title("intensity histogram")
    sum_cols_cut=np.sum(cor_matrix.T,axis=1)
    col_index=[j for j in range(len(sum_cols_cut)) ]
    plt.subplot(3,3,9),plt.plot(col_index,sum_cols_cut),plt.title("sum cols")
    # fit new data
    new_img = np.zeros((row,column))
    corrected_list=[]
    corrected_list.append(np.array([i for i in range(column)])-half_n+p_col)
    fig3 = plt.figure(figsize =(16, 9)) 
    fig3.canvas.manager.window.setWindowTitle("Line data by shift")

    for j in range(-5,5,1):
        result = np.zeros(column)
        for index in range(row):
            temp =cor_matrix[index, :]
            shift_n= shift_pixel(index,j)
            #print(f'shift pixels: {shift_n} with j={j} and index={index}')
            result = result + shift_arrray(temp,shift_n)
        header_list.append(f"intensity{j+1}")
        corrected_list.append(result)
        if j <0:
            plt.subplot(2,1,1),plt.plot(corrected_list[0],result)
            plt.title("correction via shift left-")
            plt.pause(0.1)
        else:
            plt.subplot(2,1,2),plt.plot(corrected_list[0],result)
            plt.title("correction via shift right+")
            plt.pause(0.1)
        plt.legend([i for i in range(10)])

    # save corrected_list data
    #corr_datafile=os.path.join(save_folder,f'Peakcorrected_half_n1196-2_square200-{half_n}-{filename}.xlsx')
    corr_datafile=os.path.join(save_folder,f'ROI-peakfit-{filename}.xlsx')

    corrected_data=np.array(corrected_list,dtype=np.float32).T
    # save to excel
    pd_data=pd.DataFrame(corrected_data,columns=header_list)
    #print(pd_data)

    # fit the pd_data
    x_correct=pd_data.values[:, 0]
    FWHM_results={}
    for i in range(10):
        y_correct=pd_data.values[:, i+1]
        FWHM,FWHM_err=Gaussian_FWHM(x_correct,y_correct,center=p_col,index=(i-5))
        print(f'fit-{i}:{FWHM:.4f},{FWHM_err}\n')
        FWHM_results[f'FitGauss{i}']=(FWHM,FWHM_err)
    for key,value in FWHM_results.items():
        print(f'{key}: {value}\n')
    excel_writer = pd.ExcelWriter(corr_datafile)
    pd_data.to_excel(excel_writer)
    excel_writer.save()
    print(f'save ROI peak correction to excel xlsx file successfully\nfile path: {corr_datafile}')
    plt.show()

if __name__=="__main__":
    root = Tk()
    root.withdraw()
    root.update()
    img_path = askopenfilename(title=u'Read CCD image')
    #bg_path = askopenfilename(title=u'Read background image')
    root.destroy()
    # save corrected_list data
    save_folder,file=os.path.split(img_path)
    filename,extension=os.path.splitext(file)
    print(f'save folder: {save_folder}\n filename:{filename}, type:{extension}')
    # selected point near the mid of the line
    p_col=1126
    p_row=1042
    half_n=50   # total 2*half_n rows for correction

    img = Image.open(img_path)
    matrix = np.array(img,dtype=np.float32)
    cut_matrix=matrix[:,p_col-half_n:p_col+half_n]
    Fit_peak_data(cut_matrix,p_col,half_n,save_folder,filename)