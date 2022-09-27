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

def detectorclean(exp, noise1, noise2,thresholdUP=0.95,thresholdDOWN=0.05):
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

def tif_preprocess(tif_data:np.array([])):
    """preprocess the input img data into denoised and clean_background data
    protocol:
    1. median filter
    2. dectectclean(0.95,0.05) skipped
    3. clear background
    normally is 2D data width,height=(2052,2048),peak line in width
    Args:
        tif_data (np.array): tif img 2D-array like (2052,2048)
    """
    print(f'img data shape:\n width,height={tif_data.shape}')
    median_matrix=median_filter(tif_data,filter_N=3)
    clean_matrix=detectorclean(median_matrix,noise1=50,noise2=200,thresholdUP=0.9,thresholdDOWN=0.1)
    #width,height,clearBG_matrix=clear_bg(median_matrix)
    width,height,clearBG_matrix=clear_bg(clean_matrix)
    return clearBG_matrix,median_matrix

def gaussian(x, amp, cen, wid):
    """1-d gaussian: gaussian(x, amp, cen, wid)"""
    return (amp / (sqrt(2*pi) * wid)) * exp(-(x-cen)**2 / (2*wid**2))


def GaussianFit(x,y,center=1200,info:str='GaussFit'):
    """find FWHM from the imported array data [x,y]

    Args:
        x: array
        y: array
        center: index(x) of peak
        info:data info
    Returns:
        fit_results(dict):{"cen":(value,value|None),"wid":(value,value|None),"FWHM":(value,value|None),"info":info}
    """
    Fit_results={}
    gmodel = Model(gaussian)
    result = gmodel.fit(y, x=x, amp=4006, cen=center, wid=2.6)
    #print(result.values)
    #print(result.fit_report())
    wid_fit=result.params['wid'].value
    wid_err=result.params['wid'].stderr
    cen_fit=result.params['cen'].value 
    cen_err=result.params['wid'].stderr
    Fit_results['wid']=(wid_fit,wid_err)
    Fit_results['cen']=(cen_fit,cen_err)
    Fit_results['info']=info
    if wid_err!= None:
        FWHM=wid_fit*2*np.sqrt(np.log(4))
        FWHM_err=wid_err*2*np.sqrt(np.log(4))
    else:
        FWHM=-1   # Gaussian fit failed
        FWHM_err=None
    Fit_results['FWHM']=(FWHM,FWHM_err)
    Fit_results['ini_fit']=(x,result.init_fit)
    Fit_results['best_fit']=(x,result.best_fit)
    Fit_results['origin_data']=(x,y)
    #print(f'get FWHM={FWHM:.4f} with error +/-{FWHM_err}')
    return Fit_results,FWHM

def shift_pixel(index:int,j:int=1):
    #return round(0.005 + index*0.01+index**2*(1+j*0.1)*1e-7)
    #return round((0.001+0.005*j)*index+index**2*(1e-7))
    #return round((0.005+0.0005*j)*index+index**2*(1e-7)) # best fit results0918-11
    return round((-0.0007+0.00005*j)*index+index**2*(1e-7)) # best fit results 0918-11
    #return round((-0.0006+0.00005*j)*index+index**2*(1e-7)) # best fit results 0905-12
    #return round(-(20/57.0+0.002*j+j**2*(1e-7))*index)  # for test line y=57*x-57000

def save_pd_data(pd_data: pd.DataFrame, path, filename: str):
    """
    save pandas DataForm data to excel/csv/json file by path/filename
    :param pd_data: pandas dataFrame
    :param path:
    :param filename:
    :return:
    """
    save_path = path
    if not os.path.isdir(path):
        save_path = os.getcwd()
    excel_file_path = os.path.join(save_path, filename + '.xlsx')
    # excel writer
    excel_writer = pd.ExcelWriter(excel_file_path)
    pd_data.to_excel(excel_writer)
    excel_writer.save()
    print(f'save to excel xlsx file {excel_file_path} successfully')

def cal_shift_pixel(index:int,j:int=1):
    """calculate the shift pixels 
    y=a+b*x+c*x**2  # a=0
    Args:
        index (int): _description_
        j (int, optional): _description_. Defaults to 1.

    Returns:
        shift_pixels,b_param: _description_
    """
    #b=-0.0011+0.0001*j
    b=-0.006+0.0001*j
    #b=0.004+0.0001*j
    return round(b*index+index**2*(2e-7)),b


def shift_arrray(array:np.array([]),n:int=0):
    """shift a array by n position to positive is shift left else right

    Args:
        array (np.array): 1D array data
        n (int, optional): how many position Defaults to 0.positive is shift left else right
    """
    return np.append(array[n:],array[:n])

def minimize_FWHM(peak_data:np.array([]),j_n:int=100,p_col:int=935):
    """_summary_

    Args:
        peak_data (np.array): _description_
        j_n (int, optional): _description_. Defaults to 100.
        p_col (int, optional): _description_. Defaults to 935.
    Returns:
        min_result:[FWHM,{"para":(j,b_para)},Fit_results,(x_list,result)]
    """
    row,column=peak_data.shape
    half_n=round(column/2)
    x_list=np.array([i for i in range(column)])-half_n+p_col
    j_list=[i-round(j_n/2) for i in range(j_n)]
    # find the smallest FWHM in all valid Fit_results by varying j
    FWHM_results=[]
    min_FWHM=column
    min_result=None
    cur_j=0
    cur_b_para=0
    for j in j_list:
        result = np.zeros(column)
        for index in range(row):
            temp =peak_data[index, :]
            shift_n,b_para= cal_shift_pixel(index,j)
            #print(f'shift pixels: {shift_n} with j={j} and index={index}')
            result = result + shift_arrray(temp,shift_n)
        Fit_results,FWHM=GaussianFit(x_list,result,p_col,info=(j,b_para))
        if FWHM==-1: 
            print(f'FWHM estimated failed with parameter({j},{b_para:6f}):\n{Fit_results} ')
        else:
            # Gaussian fit success
            FWHM_results.append(Fit_results)
            if FWHM<min_FWHM:
                min_FWHM=FWHM
                cur_j=int(j)
                cur_b_para=round(b_para,6)
                new_results=Fit_results
                new_linedata=(x_list,result)
                min_result=[FWHM,{"para":(cur_j,cur_b_para)},new_results,new_linedata]
            else:
                pass
    #print(f'find Gaussian fit results with minimal FWHM={min_FWHM} + \ '
    #f'and parameter: {min_result[1]["para"]}\n all results:\n{min_result}')
    return min_result,cur_j,cur_b_para

def get_corrected_img(peak_data:np.array([]),j:int=0,p_col:int=935):
    row,column=peak_data.shape
    half_n=round(column/2)
    x_list=np.array([i for i in range(column)])-half_n+p_col
    # find the  FWHM with parameter j
    FWHM_info=None
    corr_peakdata=np.zeros(shape=peak_data.shape)
    result = np.zeros(column)
    for index in range(row):
        temp =peak_data[index, :]
        shift_n,b_para= cal_shift_pixel(index,j)
        #print(f'shift pixels: {shift_n} with j={j} and index={index}')
        corr_array=shift_arrray(temp,shift_n)
        corr_peakdata[index,:]=corr_array
        result = result + corr_array
    # gauss fit 
    Fit_results,FWHM=GaussianFit(x_list,result,p_col,info=(j,b_para))
    FWHM_info=[FWHM,{"para":(j,b_para)},Fit_results,(x_list,result)]
    return corr_peakdata,FWHM_info

def partial_peak_correct(peak_data:np.array([]),slice_n:int=50,p_col:int=935,save_folder:str='./',filename:str='Tif_img'):
    """correct peak data by each slice (50 row each), add all slice results
    Args:
        peak_data (np.array): _description_
        slice_n (int, optional): _description_. Defaults to 50.
        p_col (int, optional): _description_. Defaults to 935.
    """
    row,column=peak_data.shape
    half_n=round(column/2)
    # for all slice corrected data and centered data (to p_col)
    slice_datalist=[]
    center_datalist=[]
    # for each slice corrected data
    slice_cor_headerlist=["id",'FWHM','FWHM_err',"cen","cen_err","wid","wid_err","j","b_para"]
    slice_cor_datalist=[]

    x_array=np.array([i for i in range(column)])-half_n+p_col
    slice_datalist.append(x_array)
    center_datalist.append(x_array)
    # cut the peak image into slices
    peak_slices=[]
    n_rows=int(row/slice_n)
    fulladd_array=np.zeros(column)
    for i in range(0,row,n_rows):
        peak_slices.append(peak_data[i:i+n_rows],)
    #peak_slices.pop(-1)
    for index,each_slice in enumerate(peak_slices):
        min_result,j,para_b=minimize_FWHM(each_slice,j_n=100,p_col=p_col)
        print(f'{index}-find minimal FWHM: {min_result[0]} with para=({j},{para_b})')
        plot_fit_line(min_result,index,save_folder)
        #min_result=[FWHM,{"para":(j,b_para)},Fit_results,(x_list,result)]
        slicefit_rep=min_result[-2]
        slice_cor_datalist.append([index,*slicefit_rep["FWHM"],*slicefit_rep["cen"],*slicefit_rep["wid"],*min_result[1]['para']])
        #print(f'{index}-find minimal FWHM: {min_result[0]} with para={min_result[1]["para"]}')
        # shift the fitted peak  center to p_col for futher addition
        fit_array=min_result[-2]['origin_data'][1]
        fit_center=min_result[-2]["cen"][0]
        cen_shift_n=round(fit_center-p_col)
        center_array=shift_arrray(fit_array,cen_shift_n)
        slice_datalist.append(fit_array)
        center_datalist.append(center_array)
        # all slice arrays add into one 
        fulladd_array+=center_array
    # all slice array add into one and Gauss fit 
    center_datalist.append(fulladd_array)
    #print(fulladd_array)
    Fullfit_results,fulladd_FWHM=GaussianFit(x_array,fulladd_array,p_col,info='Full addition-Gaussfit')
    Total_add_result=[fulladd_FWHM,{"para":"Full addition-Gaussfit"},Fullfit_results,(x_array,fulladd_array)]
    plot_fit_line(Total_add_result,index=slice_n,save_folder=save_folder)
    # add fullfit result
    slice_cor_datalist.append([slice_n,*Fullfit_results["FWHM"],*Fullfit_results["cen"],*Fullfit_results["wid"],0,0])
    # save to excel file

    slice_data=np.array(slice_datalist,dtype=np.float32).T
    center_data=np.array(center_datalist,dtype=np.float32).T
    
    # save to excel
    corr_pddata=pd.DataFrame(slice_cor_datalist,columns=slice_cor_headerlist)
    slice_pddata=pd.DataFrame(slice_data)
    center_pddata=pd.DataFrame(center_data)
    save_pd_data(slice_pddata,save_folder,filename=f'Gaussfit_Slice_p_col-{p_col}-{filename}')
    save_pd_data(center_pddata,save_folder,filename=f'Gaussfit_Centered_p_col-{p_col}-{filename}')
    save_pd_data(corr_pddata,save_folder,filename=f'Correction_slice_resports-{filename}')



def plot_fit_line(min_result:list,index:int=0,save_folder:str='./'):
    """ min_result:[FWHM,{"para":(j,b_para)},Fit_results,(x_list,result)]

    Args:
        min_result (list): _description_
        index (int, optional): _description_. Defaults to 0.
    """
    fig=plt.figure(figsize =(16, 9))
    fig.canvas.manager.window.setWindowTitle(f"Fit-FWHM-slice{index}")
    ax=plt.subplot()
    plt.plot(*min_result[-2]['origin_data'], 'o')
    plt.plot(*min_result[-2]['ini_fit'], '--', label='initial fit')
    plt.plot(*min_result[-2]['best_fit'], '-', label='best fit')
    FWHM_text=f'FWHM={min_result[-2]["FWHM"][0]:.4f} +/-{min_result[-2]["FWHM"][1]:.4f}'
    cen_text=f'cen={min_result[-2]["cen"][0]:.4f} +/-{min_result[-2]["cen"][1]:.4f}'
    plt.text(0.5, 0.5, s=FWHM_text+'\n'+cen_text+f'\nwith para={min_result[1]["para"]}',color = "m", transform=ax.transAxes,fontsize=15)
    plt.title(f"Best Gaussfit with minimal FWHM-Slice-{index}")
    plt.legend()
    save_fig=os.path.join(save_folder,f'Slice-{index}_fit_results.jpg')
    plt.savefig(save_fig)



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
    p_col=933
    p_row=1042
    half_n=50   # total 2*half_n rows for correction

    img = Image.open(img_path)
    img_matrix = np.array(img,dtype=np.float32)
    cut_matrix=img_matrix[:,p_col-half_n:p_col+half_n]
    clean_matrix,median_matrix=tif_preprocess(cut_matrix)

    fig2 = plt.figure(figsize =(16, 9)) 
    fig2.canvas.manager.window.setWindowTitle("image data preprocess")
    plt.subplot(3,3,1),plt.imshow(img_matrix.T,cmap=cm.rainbow,vmin=1300,vmax=1400)
    plt.colorbar(location='bottom', fraction=0.1),plt.title("cut raw image")
    plt.subplot(3,3,4),plt.hist(img_matrix.flatten(),bins=100),plt.title("intensity histogram")
    # sum columm plot
    sum_rows_cut=np.sum(img_matrix.T,axis=0)
    row_index=[i for i in range(len(sum_rows_cut)) ]
    sum_cols_cut=np.sum(img_matrix.T,axis=1)
    col_index=[j for j in range(len(sum_cols_cut)) ]
    plt.subplot(3,3,7),plt.plot(col_index,sum_cols_cut),plt.title("sum cols")
    # plot median matrix
    plt.subplot(3,3,2),plt.imshow(median_matrix.T,cmap=cm.rainbow,vmin=1300,vmax=1400)
    plt.colorbar(location='bottom', fraction=0.1),plt.title("median blur image")
    plt.subplot(3,3,5),plt.hist(median_matrix.flatten(),bins=200),plt.title("intensity histogram")
    sum_cols_cut=np.sum(median_matrix.T,axis=1)
    col_index=[j for j in range(len(sum_cols_cut)) ]
    plt.subplot(3,3,8),plt.plot(col_index,sum_cols_cut),plt.title("sum cols")
    # plot clean background img
    plt.subplot(3,3,3),plt.imshow(clean_matrix.T,cmap=cm.rainbow,vmin=-25,vmax=25),plt.title("clear background")
    plt.colorbar(location='bottom', fraction=0.1)
    plt.subplot(3,3,6),plt.hist(clean_matrix.flatten(),bins=200),plt.title("intensity histogram")
    sum_cols_cut=np.sum(clean_matrix.T,axis=1)
    col_index=[j for j in range(len(sum_cols_cut)) ]
    plt.subplot(3,3,9),plt.plot(col_index,sum_cols_cut),plt.title("sum cols")
    # find the minimal FWHM of full peak img
    min_result,j,para_b=minimize_FWHM(clean_matrix,j_n=100,p_col=p_col)
    print(f'minimal FWHM from:{min_result[0]},({j},{para_b})')
    # get the corrected img
    corr_peakdata,FWHM_info=get_corrected_img(clean_matrix,j=min_result[1]["para"][0],p_col=p_col)
    # plot the correcte img and clean peak img
    fig2 = plt.figure(figsize =(16, 9)) 
    fig2.canvas.manager.window.setWindowTitle("Compare Corrected Peak")
    plt.subplot(4,1,1),plt.imshow(clean_matrix.T,cmap=cm.rainbow,vmin=-25,vmax=25)
    plt.colorbar(location='right', fraction=0.1),plt.title("cut raw image")
    sum_cols_cut=np.sum(clean_matrix.T,axis=1)
    col_index=[j for j in range(len(sum_cols_cut)) ]
    plt.subplot(4,1,2),plt.plot(col_index,sum_cols_cut),plt.title("sum cols")
    # corrected img
    plt.subplot(4,1,3),plt.imshow(corr_peakdata.T,cmap=cm.rainbow,vmin=-25,vmax=25)
    plt.colorbar(location='right', fraction=0.1),plt.title("cut raw image")
    sum_cols_cut=np.sum(corr_peakdata.T,axis=1)
    col_index=[j for j in range(len(sum_cols_cut)) ]
    plt.subplot(4,1,4),plt.plot(col_index,sum_cols_cut),plt.title("sum cols")
    partial_peak_correct(clean_matrix,slice_n=19,p_col=p_col,save_folder=save_folder,filename=filename)
    plt.show()