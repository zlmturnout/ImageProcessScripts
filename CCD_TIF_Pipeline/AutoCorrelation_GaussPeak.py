# coding: utf-8
import os, sys,time,datetime
import traceback
import numpy as np
import matplotlib.pyplot as plt
import tkinter
from tkinter import *
from tkinter.filedialog import askopenfilename
import matplotlib.image as mpimage
from PIL import Image
import csv,cv2
from scipy import interpolate
from scipy.optimize import curve_fit
import matplotlib.cm as cm
import pandas as pd
from numpy import exp, loadtxt, pi, sqrt
from lmfit import Model

def creatPath(file_path):
    """
    create a given path if not exist and return it
    :param file_path:
    :return: file_path
    """
    if os.path.exists(file_path) is False:
        os.makedirs(file_path)
    return file_path

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

def tif_preprocess(tif_data:np.array([]),detector_clean:bool=True,cv_filter:bool=True):
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
    if cv_filter:
        median_matrix=median_filter(tif_data,filter_N=3)
    else:
        median_matrix=tif_data
    if detector_clean:
        clean_matrix=detectorclean(median_matrix,noise1=50,noise2=200,thresholdUP=0.9,thresholdDOWN=0.1)
    else:
        clean_matrix=median_matrix
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
    # # excel writer
    # excel_writer = pd.ExcelWriter(excel_file_path)
    # pd_data.to_excel(excel_writer)
    # excel_writer.save()
    with pd.ExcelWriter(excel_file_path) as writer:
        pd_data.to_excel(writer)
    print('save to excel xlsx file successfully')
    #print(f'save to excel xlsx file {excel_file_path} successfully')

def direct_addrows_FWHM(full_data:np.array([]),p_col:int=935,save_folder:str='./',filename:str='direct_addrows'):
    """find the FWHM by direct addtion of all rows

    Args:
        peak_data (np.array): _description_
        p_col (int, optional): _description_. Defaults to 935.
    """
    row,column=full_data.shape
    #print(f'row:{row},column:{column}') # row:21,column:100
    half_col=round(column/2)
    col_list=np.array([i for i in range(column)])-half_col+p_col
    # direct addition along columns
    sum_rows=np.sum(full_data,axis=0)
    Fit_results,FWHM=GaussianFit(col_list,sum_rows,center=p_col,info='direct_addrows')
    if FWHM!=-1:
        print(f'direct add rows:\npeak FWHM={Fit_results["FWHM"][0]:.4f}+/-{Fit_results["FWHM"][1]:.4f} with center at {Fit_results["cen"][0]} ')
        plot_GaussFit_results(Fit_results,save_folder,filename)
    else:
        print(f'estimate FWHM failed')
    return Fit_results

def find_peak_center(slice_data:np.array([]))->float:
    """find the Gauss peak center in each slice

    Args:
        slice_data (np.array): _description_ shape(200*2048)=(column,row)

    Returns:
        float: 
    """
    row,column=slice_data.shape
    #print(f'row:{row},column:{column}') # row:21,column:100
    half_col=round(column/2)
    col_list=np.array([i for i in range(column)])
    # direct addition along columns
    sum_rows=np.sum(slice_data,axis=0)
    Fit_results,FWHM=GaussianFit(col_list,sum_rows/row,center=half_col,info='slice sum rows fit')
    #print(f'get peak center at {Fit_results["cen"][0]} with FWHM={Fit_results["FWHM"]} ')
    if FWHM!=-1:
        return Fit_results['cen'][0]
    else:
        return half_col

def get_slice_peaks(matrix_data:np.array([]),slice_n:int=100,p_col:int=935)->tuple:
    """find the peak center (p_col) in n slice of theinput matrix_data with shape(200*2048)=(column,row)
        for line fit func:y=a+b*x+c*x^2
        a is constant
    Args:
        matrix_data (np.array): width*height=200*2048,peak line in vertical direction
        slice_n (int, optional): slices in vertical direction. Defaults to 100.
        p_col (int): center column index of the select matrix_data
    Returns:
         tuple (list,list): linepeak (x_list,y_list)
    """
    row,column=matrix_data.shape
    half_col=round(column/2)
    n_rows=int(row/slice_n)
    #row_list=[i*n_rows+round(n_rows/2) for i in range(slice_n)]
    row_list=[]
    y_list=[]
    for i in range(slice_n):
        row_list.append(i*n_rows+round(n_rows/2))
        slice_data=matrix_data[i*n_rows:(i+1)*n_rows]
        y_list.append(find_peak_center(slice_data))
    col_list=np.array(y_list)-half_col+p_col
    #print(row_list,col_list)

    return (row_list,col_list)

def peak_curve_func(x,a:float,b:float,c:float):
    return a+b*x+c*x**2

def peakline_curve_fit(x_list:np.array([]),y_list:np.array([])):
    fit_status=True
    try:
        popt, pcov = curve_fit(peak_curve_func, x_list, y_list) # 拟合方程，参数包括func，xdata，ydata，
        # 有popt和pcov两个参数，其中popt参数为a，b，c，pcov为拟合参数的协方差
    except Exception as e:
        print(traceback.format_exc()+e)
        fit_status=False
    else:
        print(f'find parameter [a,b,c]={popt}' )
        #print(f'\n with pcov={pcov}')
    return popt,fit_status

def cal_shift_pixel(index:int,p_col:int,a,b,c):
    """calculate the shift pixels 
    y=a+b*x+c*x**2  
    """
    return round(a+b*index+index**2*c-p_col)

def shift_arrray(array:np.array([]),n:int=0):
    """shift a array by n position to positive is shift left else right

    Args:
        array (np.array): 1D array data
        n (int, optional): how many position Defaults to 0.positive is shift left else right
    """
    return np.append(array[n:],array[:n])

def correlation_FWHM(peak_data:np.array([]),slice_n:int=20,p_col:int=935,save_folder:str='./',filename:str='Tif_img'):
    """find the FWHM results by correlation methods
    slices->peak center->curve-fit->shif each rows->final peak-line
    Args:
        peak_data (np.array): the selected 2D matrix contain the peak
        slice_n (int, optional): slices number. Defaults to 100.
        p_col (int, optional): center col index of the peak data . Defaults to 935.
    """
    row,column=peak_data.shape
    #print(f'peak-line 2D matrix\nrow:{row},column:{column}')
    half_n=round(column/2)
    x_list=np.array([i for i in range(column)])-half_n+p_col 
    row_list,col_list=get_slice_peaks(peak_data,slice_n=slice_n,p_col=p_col)
    pcov,fit_status=peakline_curve_fit(row_list,col_list)
    if fit_status:
        # curve fit success 
        a,b,c=pcov
        result = np.zeros(column)
        for index in range(row):
            temp =peak_data[index, :]
            shift_n=cal_shift_pixel(index,p_col,a,b,c)
            result += shift_arrray(temp,shift_n)
        # get the fianl FWHM Gaussfit results
        Fit_results,FWHM=GaussianFit(x_list,result,p_col,info="Correlation-fit")
        if FWHM==-1: 
            #print(f'FWHM estimated failed with parameter(a,b,c)={pcov}):\n{Fit_results} ')
            print(f'FWHM estimated failed with parameter (a,b,c)={pcov}):\n ')
        else:
            # Gaussian fit success
            Fit_results['para']=f'[a,b,c]=[{a:.4f},{b:.4e},{c:.4e}]'
            Fit_results['fit_para']=[a,b,c]
            print(f'get FWHM={FWHM:.4f}+/-{Fit_results["FWHM"][1]:4f} by correlation method with slice={slice_n} and p_col={p_col}')
            slice_peakdata=pd.DataFrame({"row":row_list,"center_col":col_list})
            save_pd_data(slice_peakdata,save_folder,filename=f'Gaussfit_Correlation_p_col-{p_col}-{filename}')
            #plot_GaussFit_results(Fit_results,save_folder,filename)
    return Fit_results,FWHM

def plot_GaussFit_results(Fit_results:dict,save_folder:str='./',title:str='Gaussfit with FWHM'):
    """plot the Gauss fit line and FWHM results

    Args:
        Fit_results (dict): _description_
        save_folder (str, optional): _description_. Defaults to './'.
    """
    fig=plt.figure(figsize =(16, 9))
    fig.canvas.manager.window.setWindowTitle(f'GaussFit-FWHM {Fit_results["info"]} method')
    ax=plt.subplot()
    plt.plot(*Fit_results['origin_data'], 'o',label='corrected data')
    plt.plot(*Fit_results['ini_fit'], '--', label='initial fit')
    plt.plot(*Fit_results['best_fit'], '-', label='best fit')
    FWHM_text=f'FWHM={Fit_results["FWHM"][0]:.4f} +/-{Fit_results["FWHM"][1]:.4f}'
    cen_text=f'cen={Fit_results["cen"][0]:.4f} +/-{Fit_results["cen"][1]:.4f}'
    para_text=Fit_results.get('para','') 
    plt.text(0.55, 0.5, s=FWHM_text+'\n'+cen_text+f'\n'+para_text,color = "m", transform=ax.transAxes,fontsize=15)
    plt.title(f'{Fit_results["info"]}_{title}')
    plt.legend()
    save_fig=os.path.join(save_folder,f'FWHM_{Fit_results["info"]}_{title}.jpg')
    plt.savefig(save_fig)
    
def minimal_FWHM_correlation(peak_data:np.array([]),slice_n:int=100,p_col:int=935,save_folder:str='./',filename:str='Tif_img'):
    row,column=peak_data.shape
    min_FWHM=column
    min_result=[]
    FWHM_list=[5.0,5.0]
    #slice_list=[10,20,50,100,120,180,200,250,300]
    #slice_list=[100,120,180,200,250,300]
    #slice_list=[10,20,50,100,200,250,300]
    slice_list=[10,20,50,100]
    slice_para={}
    for col_index in range(p_col-5,p_col+5):
        for slices in slice_list:
            Fit_results,FWHM=correlation_FWHM(peak_data,slice_n=slices,p_col=col_index,save_folder=save_folder,filename=filename)
            if FWHM==-1: 
                print(f'FWHM estimated failed with parameter(slice_n={slices},p_col={col_index}) ')
            else:
                FWHM_array=np.array(FWHM_list[:-1])
            # Gaussian fit success
                if FWHM<min_FWHM and FWHM>np.average(FWHM_array)*0.6:
                    min_FWHM=FWHM
                    slice_para={"slice_n":slices,"p_col":col_index}
                    min_result=[Fit_results,FWHM,slice_para]
                    FWHM_list.append(FWHM)
                else:
                    pass
    print(f'find minimal FWHM={min_result[1]:.4f} with parameter {min_result[-1]}')
    return min_result

def get_correlation_img(peak_data:np.array([]),fit_para:list,p_col:int=935,dis_const:float=29.3,vmin:int=1300,vmax:int=1380,save_folder:str='./',filename:str='correctedPeak_img',E_in:float=443.5,E_ref:float=540,Xpixel_bg_i:int=500):
    """correct raw image based on the fit_parameter of peak-line,peak center at p_col
    fit_para:[a,b,c] means y=a+b*x+c*x**2

    Args:
        peak_data (np.array): _description_
        fit_para (list): [a,b,c] y=a+b*x+c*x**2
        p_col (int, optional): peak center, Defaults to 935.
    """
    row,column=peak_data.shape
    half_n=round(column/2)
    x_list=np.array([i for i in range(column)])-half_n+p_col
    #E_out_list=-(np.array([i for i in range(column)])-p_col)*dis_const/1000+E_in
    E_out_list=-(np.array([i for i in range(column)])-p_col)*dis_const/1000+E_ref
    
    # find the  FWHM with parameter j
    FWHM_info=None
    corr_peakdata=np.zeros(shape=peak_data.shape)
    sum_result = np.zeros(column)
    [a,b,c]=fit_para
    for index in range(row):
        #temp =peak_data[index, :]
        #shift_n=cal_shift_pixel(index,p_col,a,b,c)
        #corr_array=shift_arrray(temp,shift_n)
        corr_array=peak_data[index, :]
        corr_peakdata[index,:]=corr_array
        sum_result += corr_array

    # for energy in
    E_in_list=np.array(E_in for i in range(column))
    # normalize intensity
    average_I=np.average(sum_result[Xpixel_bg_i-50:Xpixel_bg_i+50])
    NormalizeDi_I=sum_result/average_I
    NormalizeSub_I=sum_result-average_I
    NormalizeLn_I=np.log(sum_result/average_I)

    #print(corr_peakdata)
    # display corrected img
    fig, [ax1, ax2,ax3] = plt.subplots(3, 1, tight_layout=True,figsize=(16, 9))
    fig.canvas.manager.window.setWindowTitle("Display corrected image")
    im=ax1.imshow(corr_peakdata,cmap=cm.rainbow,vmin=vmin,vmax=vmax)
    fig.colorbar(im,ax=ax1,location='bottom', fraction=0.1)
    ax1.set_title("Autocorrelation corrected img")
    ax2.plot(E_out_list,sum_result, marker='o', markersize=1, markerfacecolor='orchid',
                              markeredgecolor='orchid', linestyle='-', color='c', label='corrected peak spectra')
    ax2.set_xlabel('Energy(eV)',fontsize=12, color='#20B2AA')
    ax2.set_ylabel('intensity',fontsize=12, color='#20B2AA')
    ax2.set_title(f"Sum_Corrected Spectral_{filename}")
    ax3.plot(E_out_list,NormalizeSub_I, marker='o', markersize=1, markerfacecolor='orchid',
                              markeredgecolor='orchid', linestyle='-', color='c', label='corrected peak spectra')
    ax3.set_xlabel('Energy(eV)',fontsize=12, color='#20B2AA')
    ax3.set_ylabel('intensity',fontsize=12, color='#20B2AA')
    ax3.set_title(f"Normalize_Sub_Corrected Spectral_{filename}")
    save_fig=os.path.join(save_folder,f'Autocorrelated_img_{filename}.pdf')
    plt.savefig(save_fig)

    spectra_dict={"EnergyIn(eV)":E_in_list,"EnergyOut(eV)":E_out_list,"Intensity":sum_result,"Index(pixel)":x_list,"Normalized_Di_Intensity":NormalizeDi_I,
                  "Normalized_Sub_Intensity":NormalizeSub_I,"Normalized_Ln_Intensity":NormalizeLn_I}
    pd_spectrum_data=pd.DataFrame(spectra_dict)
    save_pd_data(pd_spectrum_data,save_folder,filename=f'Corrected-FullSpectrum_E_in_{E_in}_{filename}')
    return corr_peakdata,pd_spectrum_data

if __name__=="__main__":
    root = Tk()
    root.withdraw()
    root.update()
    img_path = askopenfilename(title=u'Read CCD image')
    #bg_path = askopenfilename(title=u'Read background image')
    root.destroy()
    # save corrected_list data
    save_folder,file=os.path.split(img_path)
    corr_folder=creatPath(os.path.join(save_folder,'CorrectedResults'))
    filename,extension=os.path.splitext(file)
    print(f'save folder: {save_folder}\n filename:{filename}, type:{extension}')
    # selected point near the mid of the line
    p_col=1220
    p_row=1042
    half_n=100   # total 2*half_n rows for correction
    slice_n=50

    img = Image.open(img_path)
    img_matrix = np.array(img,dtype=np.float32)
    cut_matrix=img_matrix[:,p_col-half_n:p_col+half_n]
    clean_matrix,median_matrix=tif_preprocess(cut_matrix,False,False)

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

    # find peak-line center in  each slice and display
    slice_n=260
    row_list,col_list=get_slice_peaks(clean_matrix,slice_n=slice_n,p_col=p_col)
    fig3 = plt.figure(figsize =(16, 9)) 
    fig3.canvas.manager.window.setWindowTitle("Display slice peak center")
    plt.imshow(img_matrix,cmap=cm.rainbow,vmin=1300,vmax=1380)
    plt.colorbar(location='bottom', fraction=0.1),plt.title("slice peak center")
    plt.plot(col_list,row_list,'o',label='slice center pixel',markersize=0.5,color='b')
    

    # FWHM by direct add rows
    direct_addrows_FWHM(clean_matrix,p_col=p_col,save_folder=save_folder,filename=filename)
    # find the peak center for each slice
    slice_n=50
    #Fit_results,FWHM=correlation_FWHM(clean_matrix,slice_n=slice_n,p_col=p_col,save_folder=save_folder,filename=filename)
    #get_correlation_img(img_matrix,fit_para=Fit_results['fit_para'],p_col=p_col,save_folder=corr_folder,filename=filename)
    min_result=minimal_FWHM_correlation(clean_matrix,slice_n=slice_n,p_col=p_col,save_folder=save_folder,filename=filename)
    get_correlation_img(img_matrix,fit_para=min_result[0]['fit_para'],p_col=p_col)
    plt.show()

