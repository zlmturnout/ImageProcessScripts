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

def gaussian(x, amp, cen, wid):
    """1-d gaussian: gaussian(x, amp, cen, wid)"""
    return (amp / (sqrt(2*pi) * wid)) * exp(-(x-cen)**2 / (2*wid**2))


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
    print(f'wid_err:{wid_err}')
    FWHM=wid_fit*2*np.sqrt(np.log(4))
    if wid_err!= None:
        FWHM_err=wid_err*2*np.sqrt(np.log(4))
        FWHW_text=f'FWHM={FWHM:.4f} +/-{FWHM_err:.4f}'
    else:
        FWHM_err='estimate failed'
        FWHW_text=f'FWHM={FWHM:.4f} +/-{FWHM_err}'
    print(f'get FWHM={FWHM:.4f} with error +/-{FWHM_err}')
    fig=plt.figure(figsize =(16, 9))
    fig.canvas.manager.window.setWindowTitle(f"Fit-FWHM-{index}")
    ax=plt.subplot()
    plt.plot(x, y, 'o')
    plt.plot(x, result.init_fit, '--', label='initial fit')
    plt.plot(x, result.best_fit, '-', label='best fit')
    plt.text(0.5, 0.5, s=FWHW_text,color = "m", transform=ax.transAxes,fontsize=15)
    plt.text(0.5, 1.0, s=f'Gauss FIt-{index}',color = "m", transform=ax.transAxes,fontsize=15)
    plt.legend()
    return FWHM,FWHM_err

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
# img_path = r'D:\eline\RXES\code\data\20220102\19_38.3\TIO2@424eV_slit_1000_2000s_R1_38.3_r2.tif'
# # bg_path = r'D:\eline\REXS\code\data\20211215\-10degree_01_BGR_BGR.tif'

#img= mpimage.imread(img_path).astype('float64')
# add by limin
img = Image.open(img_path)
matrix = np.array(img,dtype=np.float32)
#background = mpimage.imread(bg_path).astype('float64')
fig1 = plt.figure(figsize =(16, 9))
fig1.canvas.manager.window.setWindowTitle("Visualize raw image")
plt.subplot(2,2,1),plt.imshow(matrix,cmap=cm.rainbow,vmin=1300,vmax=1400)
plt.colorbar(location='right', fraction=0.1),plt.title("raw image")
sum_rows_raw=np.sum(matrix,axis=0)
row_index=[i for i in range(len(sum_rows_raw)) ]
sum_cols_raw=np.sum(matrix,axis=1)
col_index=[j for j in range(len(sum_cols_raw)) ]
plt.subplot(2,2,3),plt.plot(row_index,sum_rows_raw),plt.title("sum cols")
plt.subplot(2,2,2),plt.plot(col_index,sum_cols_raw),plt.title("sum rows")

# selected point near the mid of the line
p_col=940
p_row=1042
half_n=200   # total 2*half_n rows for correction

header_list=['pixels']


cut_matrix=matrix[:,p_col-half_n:p_col+half_n].T 
fig2 = plt.figure(figsize =(16, 9)) 
fig2.canvas.manager.window.setWindowTitle("image data preprocess")
plt.subplot(3,3,1),plt.imshow(cut_matrix,cmap=cm.rainbow,vmin=1300,vmax=1400)
plt.colorbar(location='bottom', fraction=0.1),plt.title("cut raw image")
plt.subplot(3,3,4),plt.hist(cut_matrix.flatten(),bins=100),plt.title("intensity histogram")
# sum columm plot
sum_rows_cut=np.sum(cut_matrix,axis=0)
row_index=[i for i in range(len(sum_rows_cut)) ]
sum_cols_cut=np.sum(cut_matrix,axis=1)
col_index=[j for j in range(len(sum_cols_cut)) ]
plt.subplot(3,3,7),plt.plot(col_index,sum_cols_cut),plt.title("sum cols")

#matrix1 = matrix.T
filter_N=3
mean_matrix=cv2.medianBlur(cut_matrix, filter_N)
#mean_matrix = cv2.boxFilter(cut_matrix, -1, (3, 3), normalize=True)
#mean_matrix = cv2.GaussianBlur(cut_matrix, (5, 5), 1)
#matrix1 = matrix
plt.subplot(3,3,2),plt.imshow(mean_matrix,cmap=cm.rainbow,vmin=1300,vmax=1400)
plt.colorbar(location='bottom', fraction=0.1),plt.title("median blur image")
plt.subplot(3,3,5),plt.hist(mean_matrix.flatten(),bins=200),plt.title("intensity histogram")
sum_cols_cut=np.sum(mean_matrix,axis=1)
col_index=[j for j in range(len(sum_cols_cut)) ]
plt.subplot(3,3,8),plt.plot(col_index,sum_cols_cut),plt.title("sum cols")


thresholdUP = 0.9
thresholdDOWN = 0.1
matrix1 = detectorclean(mean_matrix.T, noise1=50, noise2=200)
print(type(matrix1),matrix1.shape)
row, column, out = clear_bg(matrix1) #2052*400
print(f'row: {row}\ncolumn: {column}')
plt.subplot(3,3,3),plt.imshow(out.T,cmap=cm.rainbow,vmin=-25,vmax=25),plt.title("clear background")
plt.colorbar(location='bottom', fraction=0.1)
plt.subplot(3,3,6),plt.hist(out.flatten(),bins=200),plt.title("intensity histogram")
sum_cols_cut=np.sum(out.T,axis=1)
col_index=[j for j in range(len(sum_cols_cut)) ]
plt.subplot(3,3,9),plt.plot(col_index,sum_cols_cut),plt.title("sum cols")

def shift_pixel(index:int,j:int=1):
    #return round(0.005 + index*0.005+index**2*(1+j*0.1)*1e-7)
    return round((0.001+0.005*j)*index+index**2*(1e-7))

def shift_arrray(array:np.array([]),n:int=0):
    """shift a array by n position to left if True, else right

    Args:
        array (np.array): 1D array data
        n (int, optional): how many position Defaults to 0.positive is shift left else right
    """
    return np.append(array[n:],array[:n])

cor_matrix = out
#cor_matrix = median_matrix.T
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
        result = result + shift_arrray(temp,shift_n)
    header_list.append("intensity")
    corrected_list.append(result)
    if j <0:
        plt.subplot(2,1,1),plt.plot(corrected_list[0],result)
        plt.title("correction via shift left-")
        plt.pause(0.5)
    else:
        plt.subplot(2,1,2),plt.plot(corrected_list[0],result)
        plt.title("correction via shift right+")
        plt.pause(0.5)
plt.legend([i for i in range(10)])

# save corrected_list data
#corr_datafile=os.path.join(save_folder,f'Peakcorrected_half_n1196-2_square200-{half_n}-{filename}.xlsx')
corr_datafile=os.path.join(save_folder,f'NewFit-half{half_n}_peak{p_col}-{filename}.xlsx')

corrected_data=np.array(corrected_list,dtype=np.float32).T
# save to excel
pd_data=pd.DataFrame(corrected_data,columns=header_list)
print(pd_data)

# fit the pd_data
x_correct=pd_data.values[:, 0]
FWHM_results={}
for i in range(10):
    y_correct=pd_data.values[:, i+1]
    FWHM,FWHM_err=Gaussian_FWHM(x_correct,y_correct,center=p_col,index=(1-2*int(i/5))*(i%5))
    print(f'fit-{i}:{FWHM:.4f},{FWHM_err}\n')
    FWHM_results[f'FitGauss{i}']=(FWHM,FWHM_err)
for key,value in FWHM_results.items():
    print(f'{key}: {value}\n')
excel_writer = pd.ExcelWriter(corr_datafile)
pd_data.to_excel(excel_writer)
excel_writer.save()
print(f'save to excel xlsx file successfully\nfile path: {corr_datafile}')
plt.show()

