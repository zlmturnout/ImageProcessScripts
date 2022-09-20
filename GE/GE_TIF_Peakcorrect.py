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


def Gaussian_FWHM(x,y,center=1200):
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
    fig.canvas.manager.window.setWindowTitle("Fit-FWHM")
    ax=plt.subplot()
    plt.plot(x, y, 'o')
    plt.plot(x, result.init_fit, '--', label='initial fit')
    plt.plot(x, result.best_fit, '-', label='best fit')
    plt.text(0.5, 0.5, s=FWHW_text,color = "m", transform=ax.transAxes,fontsize=15)
    plt.legend()
    return FWHM,FWHM_err


root = Tk()
root.withdraw()
root.update()
img_path = askopenfilename(title=u'Read CCD image')
root.destroy()

# save corrected data
save_folder,file=os.path.split(img_path)
filename,extension=os.path.splitext(file)
print(f'save folder: {save_folder}\n filename:{filename}, type:{extension}')

# open the original raw img
img = Image.open(img_path)
matrix = np.array(img,dtype=np.float32)
print(f'shape of the raw img:{matrix.shape}')
# plot the original raw img
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
p_col=1126
p_row=1042
half_n=200   # total 2*half_n rows for correction
'''
select (half_n=100) rows near the line at both side
'''
cut_matrix=matrix[:,p_col-half_n:p_col+half_n].T 

# plot the cut img  cut_img= from line-100 to line+100
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

#apply median filter
filter_N=3
median_matrix=median_filter(cut_matrix,filter_N)

# plot median filter
plt.subplot(3,3,2),plt.imshow(median_matrix,cmap=cm.rainbow,vmin=1300,vmax=1400)
plt.colorbar(location='bottom', fraction=0.1),plt.title("median blur image")
plt.subplot(3,3,5),plt.hist(median_matrix.flatten(),bins=200),plt.title("intensity histogram")
sum_cols_cut=np.sum(median_matrix,axis=1)
col_index=[j for j in range(len(sum_cols_cut)) ]
plt.subplot(3,3,8),plt.plot(col_index,sum_cols_cut),plt.title("sum cols")

# detect clean
thresholdUP = 0.9
thresholdDOWN = 0.1
clean_matrix = detectorclean(median_matrix.T, noise1=50, noise2=200)
print(type(clean_matrix),clean_matrix.shape)
m, n, out = clear_bg(clean_matrix)
#m, n, out = clear_bg(median_matrix.T)
print(f'row: {m}\ncolumn: {n}')

# plot the img after median filter and background clean 
plt.subplot(3,3,3),plt.imshow(out.T,cmap=cm.rainbow,vmin=-25,vmax=25),plt.title("clear background")
plt.colorbar(location='bottom', fraction=0.1)
plt.subplot(3,3,6),plt.hist(out.flatten(),bins=200),plt.title("intensity histogram")
sum_cols_cut=np.sum(out.T,axis=1)
col_index=[j for j in range(len(sum_cols_cut)) ]
plt.subplot(3,3,9),plt.plot(col_index,sum_cols_cut),plt.title("sum cols")
#plt.show()

# line correction

def shift_pixel(index:int,j:int=1):
    #return round(0.005 + index*0.01+index**2*(1+j*0.1)*1e-7)
    #return round((0.001+0.005*j)*index+index**2*(1e-7))
    #return round((0.015+0.0005*j)*index+index**2*(1e-7)) # best fit results0918-11
    return round((0.006+0.0005*j)*index+index**2*(1e-7)) # best fit results 0905-12
    #return round(-(20/57.0+0.002*j+j**2*(1e-7))*index)  # for test line y=57*x-57000
    

header_list=['pixels']

cor_matrix = out
#cor_matrix = median_matrix.T
new_img = np.zeros((m,n))

index=half_n
corrected_list=[]

# left move rows, dd = shift_pixel(ii,j)
fig3 = plt.figure(figsize =(16, 9))
fig3.canvas.manager.window.setWindowTitle("Visualize peak correction")
low_lim = round(index*20 -1200)
high_lim = round(index*20 + 1200)
xinitial = np.arange(n)
corrected_list.append(xinitial[round(low_lim/20):round(high_lim/20)]-half_n+p_col)
for j in range(0,5,1):
    low_lim = round(index*20 -1200)
    high_lim = round(index*20 + 1200)
    new_X = np.arange(low_lim, high_lim)
    result = np.zeros(len(new_X))
    xinitial = np.arange(n)
    xinterp = np.arange(0, n, 0.05)
    xx = np.linspace(0, len(xinterp), n)
    for ii in range(m):
        temp =cor_matrix[ii, :]
        ntemp = fastinterp1(xinitial, temp, xinterp)
        dd= shift_pixel(ii,j)
        #print(f'shift pixels: {dd} with j={j} and index={ii}')
        #new_img[ii,:] = fastinterp1(new_X,ntemp[low_lim-dd:high_lim-dd],xx)
        result = result + ntemp[low_lim-dd:high_lim-dd]
    #print(result)
    
    y_ = fastinterp1(new_X, result, xx)
    # append data
    
    corrected_list.append(y_[round(low_lim/20):round(high_lim/20)])
    header_list.append(f"intensity{j+1}")
    plt.subplot(2,1,1),plt.plot(xinitial[round(low_lim/20):round(high_lim/20)]-half_n+p_col,
        y_[round(low_lim/20):round(high_lim/20)])
    plt.title("correction via shift right+")
    plt.pause(0.5)
    plt.legend([i for i in range(10)])

# right move rows, dd = shift_pixel(ii,-j)
for j in range(0,5,1):
    low_lim = round(index*20 -1200)
    high_lim = round(index*20 + 1200)
    new_X = np.arange(low_lim, high_lim)
    result = np.zeros(len(new_X))
    xinitial = np.arange(n)
    xinterp = np.arange(0, n, 0.05)
    xx = np.linspace(0, len(xinterp), n)
    for ii in range(m):
        temp =cor_matrix[ii, :]
        ntemp = fastinterp1(xinitial, temp, xinterp)
        dd= shift_pixel(ii,-j)
        #print(f'shift pixels: {dd} with j={j} and index={ii}')
        #new_img[ii,:] = fastinterp1(new_X,ntemp[low_lim-dd:high_lim-dd],xx)
        result = result + ntemp[low_lim-dd:high_lim-dd]
    print(result)
    y_ = fastinterp1(new_X, result, xx)
    # append data
    corrected_list.append(y_[round(low_lim/20):round(high_lim/20)])
    header_list.append(f"intensity_-{j+1}")
    plt.subplot(2,1,2),plt.plot(xinitial[round(low_lim/20):round(high_lim/20)]-half_n+p_col,
        y_[round(low_lim/20):round(high_lim/20)])
    plt.title("correction via shift left-")
    plt.pause(0.1)
plt.legend([i for i in range(10)])

# save corrected_list data
#corr_datafile=os.path.join(save_folder,f'Peakcorrected_half_n1196-2_square200-{half_n}-{filename}.xlsx')
corr_datafile=os.path.join(save_folder,f'Peakcorrected_testline-{half_n}-{filename}.xlsx')

corrected_data=np.array(corrected_list,dtype=np.float32).T
# save to excel
pd_data=pd.DataFrame(corrected_data,columns=header_list)
print(pd_data)

# fit the pd_data
x_correct=pd_data.values[:, 0]
FWHM_results={}
for i in range(10):
    y_correct=pd_data.values[:, i+1]
    FWHM,FWHM_err=Gaussian_FWHM(x_correct,y_correct,center=p_col)
    print(f'fit-{i}:{FWHM:.4f},{FWHM_err}\n')
    FWHM_results[f'Fit-{i}']=(FWHM,FWHM_err)
for key,value in FWHM_results.items():
    print(f'{key}: {value}\n')
excel_writer = pd.ExcelWriter(corr_datafile)
pd_data.to_excel(excel_writer)
excel_writer.save()
print(f'save to excel xlsx file successfully\nfile path: {corr_datafile}')
plt.show()

