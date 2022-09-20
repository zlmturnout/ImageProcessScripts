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


def Gaussian_FWHM(pd_data,center=1277):
    """find FWHM from the imported pd_data [x,y]

    Args:
        pd_data (_type_): _description_
    """
    x = pd_data.values[:, 0]
    y = pd_data.values[:, 1]
    gmodel = Model(gaussian)
    result = gmodel.fit(y, x=x, amp=1, cen=center, wid=2.6)
    print(result.values)
    print(result.fit_report())
    wid_fit=result.params['wid'].value
    wid_err=result.params['wid'].stderr
    FWHM=wid_fit*2*np.sqrt(np.log(4))
    if wid_err:
        FWHM_err=wid_err*2*np.sqrt(np.log(4))
    else:
        FWHM_err='Gauss fit failed'
    print(f'get FWHM={FWHM:.4f} with error +/-{FWHM_err}')
    FWHW_text=f'FWHM={FWHM:.4f} +/-{FWHM_err}'
    fig=plt.figure(figsize =(16, 9))
    fig.canvas.manager.window.setWindowTitle("Gaussian fit-raw estimate")
    ax=plt.subplot()
    plt.plot(x, y, 'o')
    plt.plot(x, result.init_fit, '--', label='initial fit')
    plt.plot(x, result.best_fit, '-', label='best fit')
    plt.text(0.5, 0.5, s=FWHW_text,color = "m", transform=ax.transAxes,fontsize=15)
    plt.legend()

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

# selected point near the mid of the line
p_col=1126
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
m, n, out = clear_bg(matrix1)
print(f'row: {m}\ncolumn: {n}')
plt.subplot(3,3,3),plt.imshow(out.T,cmap=cm.rainbow,vmin=-25,vmax=25),plt.title("clear background")
plt.colorbar(location='bottom', fraction=0.1)
plt.subplot(3,3,6),plt.hist(out.flatten(),bins=200),plt.title("intensity histogram")
sum_cols_cut=np.sum(out.T,axis=1)
col_index=[j for j in range(len(sum_cols_cut)) ]
plt.subplot(3,3,9),plt.plot(col_index,sum_cols_cut),plt.title("sum cols")
#plt.show()

# line correction
#matrix2 = out
#cor_matrix = mean_matrix.T
cor_matrix = out
new_img = np.zeros((m,n))
'''
plt.subplot(1,1,1)
plt.imshow(matrix)
plt.show()
'''
index=half_n
corrected_list=[]
# left move rows, dd = round(k*ii)
fig3 = plt.figure(figsize =(16, 9))
fig3.canvas.manager.window.setWindowTitle("Visualize peak correction")
low_lim = round(index*20 -1200)
high_lim = round(index*20 + 1200)
xinitial = np.arange(n)
corrected_list.append(xinitial[round(low_lim/20):round(high_lim/20)]-half_n+p_col)

def shift_pixel(index:int,j:int=1):
    #return round(0.005 + index*0.005+index**2*(1+j*0.1)*1e-7)
    return round((0.001+0.005*j)*index+index**2*(1e-7))

for j in range(0,5,1):
    #k = 0.4 + j*0.1+j**2*(1e-07)
    #k = 0.05 + j*0.1+j**2*(1e-07)
    #k = 0.01 + j*0.01+j**2*(1e-07)
    #k = 0.01 + j*0.01+j**2*(1e-07)
    k = 0.01 + j*0.01+j**2*(1e-07)
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
        temp =cor_matrix[ii, :]
        ntemp = fastinterp1(xinitial, temp, xinterp)
        #dd = round(k*ii)
        dd= shift_pixel(ii,j)
        new_img[ii,:] = fastinterp1(new_X,ntemp[low_lim-dd:high_lim-dd],xx)
        #f=interpolate.interp1d(new_X,ntemp[low_lim-dd:high_lim-dd],kind='slinear')
        #new_img = f(xx)
        result = result + ntemp[low_lim-dd:high_lim-dd]
    #print(result)
    
    y_ = fastinterp1(new_X, result, xx)
    # append data
    
    corrected_list.append(y_[round(low_lim/20):round(high_lim/20)])
    header_list.append("intensity")
    plt.subplot(2,1,1),plt.plot(xinitial[round(low_lim/20):round(high_lim/20)]-half_n+p_col,
        y_[round(low_lim/20):round(high_lim/20)])
    plt.title("correction via shift right+")
    plt.pause(0.5)
    plt.legend([i for i in range(10)])

# right move rows, dd = -round(k*ii)
for j in range(0,5,1):
    #k = 0.4 + j*0.1+j**2*(1e-07)
    #k = 0.05 + j*0.1+j**2*(1e-07)
    k = 0.00 + j*0.01+0*j**2*(1e-07)
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
        temp =cor_matrix[ii, :]
        ntemp = fastinterp1(xinitial, temp, xinterp)
        #dd = -round(k*ii)
        dd= shift_pixel(ii,j*-1)
        new_img[ii,:] = fastinterp1(new_X,ntemp[low_lim-dd:high_lim-dd],xx)
        #f=interpolate.interp1d(new_X,ntemp[low_lim-dd:high_lim-dd],kind='slinear')
        #new_img = f(xx)
        result = result + ntemp[low_lim-dd:high_lim-dd]
    print(result)
    y_ = fastinterp1(new_X, result, xx)
    # append data
    corrected_list.append(y_[round(low_lim/20):round(high_lim/20)])
    header_list.append("intensity")
    plt.subplot(2,1,2),plt.plot(xinitial[round(low_lim/20):round(high_lim/20)]-half_n+p_col,
        y_[round(low_lim/20):round(high_lim/20)])
    plt.title("correction via shift left-")
    plt.pause(0.5)
plt.legend([i for i in range(10)])

# save corrected_list data
corr_datafile=os.path.join(save_folder,f'New03corrected-{filename}.xlsx')

corrected_data=np.array(corrected_list,dtype=np.float32).T
# save to excel
pd_data=pd.DataFrame(corrected_data,columns=header_list)
print(pd_data)
# fit the pd_data
Gaussian_FWHM(pd_data,center=p_col)
excel_writer = pd.ExcelWriter(corr_datafile)
pd_data.to_excel(excel_writer)
excel_writer.save()
print(f'save to excel xlsx file successfully\nfile path: {corr_datafile}')
plt.show()

