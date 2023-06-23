# coding: utf-8
import os, sys,time,datetime
import traceback
import numpy as np
import matplotlib.pyplot as plt
import tkinter
from tkinter import *
from tkinter.filedialog import askopenfilename
import matplotlib.image as mpimage
from matplotlib.ticker import MaxNLocator
from PIL import Image
import csv,cv2
from scipy import interpolate
from scipy.optimize import curve_fit
import matplotlib.cm as cm
import pandas as pd
from numpy import exp, loadtxt, pi, sqrt
from lmfit import Model

def createPath(file_path):
    """
    create a given path if not exist and return it
    :param file_path:
    :return: file_path
    """
    if os.path.exists(file_path) is False:
        os.makedirs(file_path)
    return file_path

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
    with pd.ExcelWriter(excel_file_path) as writer:
        pd_data.to_excel(writer)
    #print(f'save to excel xlsx file {excel_file_path} successfully')

def save_tif_data(img_matrix:np.array([]),path:str,filename:str="tif_data"):
    """save img date to tif

    Args:
        img_matrix (np.array): _description_
        path (str): _description_
        filename (str, optional): _description_. Defaults to "tif_data".
    """
    save_path = path if os.path.isdir(path) else os.getcwd()
    if img_matrix.size!=0:
        pil_image=Image.fromarray(img_matrix)
        new_tif=os.path.join(save_path,f"{filename}.tif")
        pil_image.save(new_tif)
    
    #Gauss fit funcs
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

# curve fitting funcs --secondary curves
def peak_curve_func(x,a:float,b:float,c:float):
    return a+b*x+c*x**2

def peakline_curve_fit(x_list:np.array([]),y_list:np.array([])):
        fit_status=False
        try:
            popt, pcov = curve_fit(peak_curve_func, x_list, y_list) # 拟合方程，参数包括func，xdata，ydata，
            # 有popt和pcov两个参数，其中popt参数为a，b，c，pcov为拟合参数的协方差
        except Exception as e:
            print(traceback.format_exc()+e)
            fit_status=False
        else:
            fit_status=True
            #print(f'find parameter [a,b,c]={popt}' )
            #print(f'\n with pcov={pcov}')
        return popt,fit_status


    
class TifAutoCorrelation(object):
    """pipline to process CCD tif images and fit the peak line (2st order curve) 
    1. select the ROI with peak inside to the shape=(cut_lines,height)
    2. image preprocess == clear background noise  and median filter
    3. AutoCorrelationProcess == slices->peak center->curve-fit->shif each rows->final peak-line
    4. obtain and save corrected image == peak line normalized to v-line
    Needed input: tif image with shape=(width,height) like (2052,2048) with 16bit data
    Args:
        object (_type_): _description_
    """

    def __init__(self,E_ref:int=450,E_ref_col:int=1200,dis_const:float=29.3) -> None:
        super(TifAutoCorrelation,self).__init__()
        self.save_path=os.getcwd()
        self.raw_tif_data=np.array([])
        self.raw_bg_data=np.array([])
        self.fitData_folder=self.save_path
        self.file_title='Tif_AutoCorrelation'
        self.curve_fit_paras=[1201,1.72e-02,8.77e-08] # initialized a 2st-order curve fit parameter:y=a+b*x+c*x**2
        self.dis_const=dis_const  # dispersion constant 29.3meV/pixel
        self.E_ref=E_ref # reference elastic energy 450eV
        self.min_p_col=E_ref_col # reference elastic energy 450eV at column index =1200
    # **************************************LIMIN_Zhou_at_SSRF_BL20U**************************************
    """
    tif data part
    """
    def input_tif_data(self,tif_file:str):
        """get the original tif data 

        Args:
            tif_path (str): _description_
        Returns:
            raw_data,file_title
        """
        if os.path.isfile(tif_file) and tif_file.endswith('.tif'):
            # obtain the save folder info
            tif_path,file=os.path.split(tif_file)
            self.file_title,extension=os.path.splitext(file)
            self.save_path=createPath(os.path.join(tif_path,f"ACorr-{self.file_title}"))
            self.fitData_folder=createPath(os.path.join(self.save_path,'CorrectedResults'))
            img = Image.open(tif_file)
            self.raw_tif_data = np.array(img,dtype=np.float32)
            print(f'shape of the read img={np.shape(self.raw_tif_data)}')
        return self.raw_tif_data,self.file_title
    
    def input_bg_data(self,bg_file:str):
        """input background data

        Args:
            bg_file (str): _description_
        """
        if os.path.isfile(bg_file) and bg_file.endswith('.tif'):
            # obtain the save folder info
            tif_path,file=os.path.split(bg_file)
            self.bg_file_title,extension=os.path.splitext(file)
            img = Image.open(bg_file)
            self.raw_bg_data = np.array(img,dtype=np.float32)
            print(f'shape of the read img={np.shape(self.raw_bg_data)}')
        return self.raw_bg_data,self.bg_file_title
    
    def get_ROI_data(self,p_col:int=1200,cut_lines:int=50):
        """cut the image with peak inside to the shape=(cut_lines,height)

        Args:
            p_col (int, optional): center column index of peak. Defaults to 1200.
            cut_lines (int, optional): columns to be cut Defaults to 50.
        """
        self.ROI_matrix=np.array([])
        half_n=round(cut_lines/2)
        if self.raw_tif_data.size!=0:
            w,h=self.raw_tif_data.shape
            start_col=p_col-half_n if p_col>half_n else 0
            end_col=p_col+half_n if w>p_col+half_n else w-1
            self.ROI_matrix=self.raw_tif_data[:,start_col:end_col]
        return self.ROI_matrix
    # **************************************LIMIN_Zhou_at_SSRF_BL20U**************************************
    """
    end of tif data part
    """

    # **************************************LIMIN_Zhou_at_SSRF_BL20U**************************************
    """
    tif data preprocess part
    """ 
    @staticmethod
    def detectorclean(exp, noise1, noise2,thresholdUP=0.95,thresholdDOWN=0.05):
        exp = exp - np.mean(exp[:, noise1:noise2])
        exp[exp > (np.max(exp) * thresholdUP)] = 0
        exp[exp < (np.min(exp) * thresholdDOWN)] = 0
        detectorcleanout = exp
        return detectorcleanout
    
    @staticmethod
    def median_filter(matrix:np.array([]),filter_N:int=3):
        median_matrix=cv2.medianBlur(matrix, filter_N)
        return median_matrix

    @staticmethod
    def clear_bg(tif_data:np.array([]),n:int=10):
        """clean the background data
        by linear substraction 

        Args:
            tif_data (np.array): _description_
            n (int): n lines at two edge for background substraction
        Returns:
            _type_: _description_
        """
        w, h = tif_data.shape
        clean_data = np.zeros((w, h))
        for i in np.arange(w):
            k = (np.sum(tif_data[i, 1:n]) - np.sum(tif_data[i, -n:-1]))/(h)/n
            b = np.sum(tif_data[i, 1:n])/n - k*n
            exp_bg = -k * np.arange(h) +b+tif_data[i, 0]/n
            clean_data[i, :] = tif_data[i, :] - exp_bg
        return w, h, clean_data

    def tif_preprocess(self,tif_data:np.array([]),detector_clean:bool=False,cv_filter:bool=True,clean_bg:bool=True):
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
            median_matrix=self.median_filter(tif_data,filter_N=3)
        else:
            median_matrix=tif_data
        if detector_clean:
            clean_matrix=self.detectorclean(median_matrix,noise1=50,noise2=200,thresholdUP=0.9,thresholdDOWN=0.1)
        else:
            clean_matrix=median_matrix
            #width,height,clearBG_matrix=clear_bg(median_matrix)
        if clean_bg:
            width,height,clearBG_matrix=self.clear_bg(clean_matrix)
        else:
            clearBG_matrix=clean_matrix
        # input ROI data for autoCorrelation process
        self.clean_ROI_data=clearBG_matrix
        return clearBG_matrix,median_matrix

    # **************************************LIMIN_Zhou_at_SSRF_BL20U**************************************
    """
    end of tif data preprocess part
    """ 
    
    # **************************************LIMIN_Zhou_at_SSRF_BL20U**************************************
    """
     autocorrelation preprocess part
    """ 
    @staticmethod
    def cal_shift_pixel(index:int,p_col:int,fit_para:list):
        """calculate the shift pixels 
        y=a+b*x+c*x**2  
        """
        [a,b,c]=fit_para
        return round(peak_curve_func(index,a,b,c)-p_col)
    
    @staticmethod
    def shift_arrray(array:np.array([]),n:int=0):
        """shift a array by n position to positive is shift left else right

        Args:
            array (np.array): 1D array data
            n (int, optional): how many position Defaults to 0.positive is shift left else right
        """
        return np.append(array[n:],array[:n])
    
    @staticmethod
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

    def get_slice_peaks(self,matrix_data:np.array([]),slice_n:int=100,p_col:int=935)->tuple:
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
            y_list.append(self.find_peak_center(slice_data))
        col_list=np.array(y_list)-half_col+p_col
        #print(row_list,col_list)
        return (row_list,col_list)

    def correlation_FWHM(self,peak_data:np.array([]),slice_n:int=20,p_col:int=935,filename:str='Tif_img'):
        """find the FWHM results by correlation methods
        slices->peak center->curve-fit->shif each rows->final peak-line
        Args:
            peak_data (np.array): the selected 2D matrix contain the peak
            slice_n (int, optional): slices number. Defaults to 100.
            p_col (int, optional): center col index of the peak data . Defaults to 935.
        Returns:
            Fit_results,FWHM:Fit_results={'fit_para':[a,b,c],'para':[a,b,c]=[a,b,c]}
        """
        row,column=peak_data.shape
        #print(f'peak-line 2D matrix\nrow:{row},column:{column}')
        half_n=round(column/2)
        x_list=np.array([i for i in range(column)])-half_n+p_col 
        row_list,col_list=self.get_slice_peaks(peak_data,slice_n=slice_n,p_col=p_col)
        pcov,fit_status=peakline_curve_fit(row_list,col_list)
        if fit_status:
            # curve fit success 
            a,b,c=pcov
            result = np.zeros(column)
            for index in range(row):
                temp =peak_data[index, :]
                shift_n=self.cal_shift_pixel(index,p_col,self.curve_fit_paras)
                result += self.shift_arrray(temp,shift_n)
            # get the fianl FWHM Gaussfit results
            Fit_results,FWHM=GaussianFit(x_list,result,p_col,info="Correlation-fit")
            if FWHM==-1: 
                #print(f'FWHM estimated failed with parameter(a,b,c)={pcov}):\n{Fit_results} ')
                print(f'FWHM estimated failed with parameter (a,b,c)={pcov}):\n ')
            else:
                # Gaussian fit success
                Fit_results['para']=f'[a,b,c]=[{a:.4f},{b:.4e},{c:.4e}]'
                Fit_results['fit_para']=[a,b,c]
                self.curve_fit_paras=[a,b,c]
                #print(f'get FWHM={FWHM:.4f}+/-{Fit_results["FWHM"][1]:4f} by correlation method with slice={slice_n} and p_col={p_col}')
                slice_peakdata=pd.DataFrame({"row":row_list,"center_col":col_list})
                save_pd_data(slice_peakdata,self.fitData_folder,filename=f'Gaussfit_Correlation_p_col-{p_col}-{filename}')
                #plot_GaussFit_results(Fit_results,save_folder,filename)
        return Fit_results,FWHM
 
    def minimal_FWHM_correlation(self,peak_data:np.array([]),slice_n:int=100,p_col:int=935):
        """_summary_

        Args:
            peak_data (np.array): _description_
            slice_n (int, optional): _description_. Defaults to 100.
            p_col (int, optional): _description_. Defaults to 935.

        Returns:
            min_result:[Fit_results,FWHM,slice_para]
        """
        print(f"Start Run AutoCorrelation and Gaussfit process ......\n")
        row,column=peak_data.shape
        min_FWHM=column
        min_result=[]
        FWHM_list=[5.0,5.0] # to eliminate too small or too large FWHM (unreasonable Gaussfit)
        #slice_list=[10,20,50,100,120,180,200,250,300]
        #slice_list=[100,120,180,200,250,300]
        #slice_list=[10,20,50,100,200,250,300]
        slice_list=[10,20,50,100]
        slice_para={}
        for col_index in range(p_col-5,p_col+5):
            for slices in slice_list:
                Fit_results,FWHM=self.correlation_FWHM(peak_data,slice_n=slices,p_col=col_index,filename=f'slice-{slices}')
                if FWHM==-1: 
                    print(f'FWHM estimated failed with parameter(slice_n={slices},p_col={col_index}) ')
                else:
                    FWHM_array=np.array(FWHM_list[:-1])
                # Gaussian fit success
                    if FWHM<min_FWHM and FWHM>np.average(FWHM_array)*0.6:
                        min_FWHM=FWHM
                        slice_para={"slice_n":slices,"p_col":col_index}
                        self.min_p_col=col_index
                        min_result=[Fit_results,FWHM,slice_para]
                        FWHM_list.append(FWHM)
                    else:
                        pass
        print(f'find minimal FWHM={min_result[1]:.4f} with parameter {min_result[-1]}')
        return min_result

    # **************************************LIMIN_Zhou_at_SSRF_BL20U**************************************
    """
     end of autocorrelation preprocess part
    """
    
    # **************************************LIMIN_Zhou_at_SSRF_BL20U**************************************
    """
     corrected image part
    """

    def colIndex_To_Energy(self,col:int):
        dis_const=self.dis_const
        E_ref=self.E_ref
        p_col=self.min_p_col
        return E_ref-(col-p_col)*dis_const/1000

    def Energy_To_colIndex(self,E:int):
        dis_const=self.dis_const
        E_ref=self.E_ref
        p_col=self.min_p_col
        return (E_ref-E)*1000/dis_const-p_col

    def get_correct_peak_data(self,peak_data:np.array([]),fit_para:list,p_col:int=935):
        """correct raw image based on the fit_parameter of peak-line,peak center at p_col
        fit_para:[a,b,c] means y=a+b*x+c*x**2

        Args:
            peak_data (np.array): _description_
            fit_para (list): [a,b,c] y=a+b*x+c*x**2
            p_col (int, optional): peak center, Defaults to 935.
        """
        row,column=peak_data.shape
        corr_peakdata=np.zeros(shape=peak_data.shape)
        for index in range(row):
            temp =peak_data[index, :]
            shift_n=self.cal_shift_pixel(index,p_col,fit_para)
            corr_array=self.shift_arrray(temp,shift_n)
            #corr_array=peak_data[index, :]
            corr_peakdata[index,:]=corr_array
        # save corrected peak data
        self.correct_peak_data=corr_peakdata
        return corr_peakdata

    def get_spectral_info(self,correct_data:np.array([]),bg_data:np.array([]),E_in:float=443.5,E_ref:float=450,p_col:int=935,
                            dis_const:float=29.3,filename:str='correctedPeak_img',):
        """get the spectral info based on the corrected img data and normalized spectral data

        Args:
            correct_data (np.array): _description_
            fit_para (list): curve fit parameter
            E_in (float, optional): Energy in Defaults to 443.5.
            E_ref (float, optional): reference Energy Defaults to 450.
            p_col (int, optional): peak center to reference Energy Defaults to 935.
            dis_const (float, optional): dispersion constant. Defaults to 29.3.
            filename (str, optional): _description_. Defaults to 'correctedPeak_img'.
            Xpixel_bg_i (int, optional): backgroud pixel index for Normalization Defaults to 500.
        """
        row,column=correct_data.shape
        
        half_n=round(column/2)
        x_list=np.array([i for i in range(column)])-half_n+p_col
        E_out_list=-(np.array([i for i in range(column)])-p_col)*dis_const/1000+E_ref
        sum_result=np.sum(correct_data,axis=0) # sum spectral
        # energy in list
        E_in_list=np.array(E_in for i in range(column))
        
        #  calculate background intensity
        half_row=round(row/2)
        bg_lines=round(row*0.25)+1 # 50% lines around the center row index=bg_index
        #bg_index=Xpixel_bg_i if Xpixel_bg_i>bg_lines and Xpixel_bg_i<column-bg_lines else bg_lines
        # background
        if bg_data.size!=0 and bg_data.shape[0]==row:
            average_I=np.average(bg_data[half_row-bg_lines:half_row+bg_lines],axis=0)
        else:
            average_I=np.average(correct_data[half_row-bg_lines:half_row+bg_lines])
        # normalize intensity
        #average_I=np.average(sum_result[bg_index-bg_lines:bg_index+bg_lines])
        NormalizeDi_I=sum_result/average_I/row
        NormalizeSub_I=sum_result-average_I*row
        NormalizeLn_I=np.log(sum_result/average_I/row)
        # save the spectral data
        spectra_dict={"EnergyIn(eV)":E_in_list,"EnergyOut(eV)":E_out_list,"Intensity":sum_result,"Index(pixel)":x_list,"Normalized_Di_Intensity":NormalizeDi_I,
                  "Normalized_Sub_Intensity":NormalizeSub_I,"Normalized_Ln_Intensity":NormalizeLn_I}
        pd_spectrum_data=pd.DataFrame(spectra_dict)
        save_pd_data(pd_spectrum_data,self.save_path,filename=f'Corrected-FullSpectrum_E_in_{E_in}_{filename}')
        # save corrected img
        save_tif_data(correct_data,self.save_path,f'CorrectedPeak_{filename}')
        return pd_spectrum_data

    def plot_corrected_data(self,raw_data:np.array([]),ROI_data:np.array([]),corr_peakdata:np.array([]),
                            ROI_correct_data:np.array([]),pd_spectral_data:pd.DataFrame,filename:str='Autocorrelated_img'):
        """plot the raw data and auto corrected data and spectral info

        Args:
            raw_data (np.array): _description_
            correct_data (np.array): _description_
            pd_spectral_data (pd.DataFrame): _description_
        """
        
        vmin=np.min(raw_data)
        vmax=np.average(self.median_filter(raw_data))+100
        # display corrected img
        #fig, axs = plt.subplots(3, 2, tight_layout=True,figsize=(16, 9))
        fig=plt.figure(tight_layout=True,figsize=(16, 9))
        fig.canvas.manager.window.setWindowTitle("Display raw and corrected image")
        # RAW image
        raw_ax=plt.subplot(421)
        raw_im=raw_ax.imshow(raw_data,cmap=cm.rainbow,vmin=vmin,vmax=vmax)
        raw_ax.tick_params(top=False, labeltop=True, bottom=True, labelbottom=False)
        #fig.colorbar(raw_im,ax=raw_ax,location='right', fraction=0.1)
        raw_ax.set_title("Raw img")
        # corrected data
        corr_ax=plt.subplot(422)
        im=corr_ax.imshow(corr_peakdata,cmap=cm.rainbow,vmin=vmin,vmax=vmax)
        corr_ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        secax = corr_ax.secondary_xaxis('bottom', functions=(self.colIndex_To_Energy, self.Energy_To_colIndex))
        secax.set_xlabel('Energy/eV')
        secax.xaxis.set_major_locator(MaxNLocator(5)) 
        fig.colorbar(im,ax=corr_ax,location='right', fraction=0.1)
        corr_ax.set_title("Autocorrelation corrected img")

        # peak ROI data
        ROI_ax=plt.subplot(412)
        raw_im=ROI_ax.imshow(ROI_data.T,cmap=cm.rainbow,vmin=vmin,vmax=vmax)
        ROI_ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        ROI_ax.set_title("ROI Peak area")
        # correct peak ROI data
        ROI_cor_ax=plt.subplot(413)
        raw_im=ROI_cor_ax.imshow(ROI_correct_data.T,cmap=cm.rainbow,vmin=vmin,vmax=vmax)
        ROI_cor_ax.tick_params(top=False, labeltop=True, bottom=True, labelbottom=False)
        ROI_cor_ax.set_title("correct ROI Peak area")
        
        # spectral data
        E_out_list=pd_spectral_data['EnergyOut(eV)']
        sum_result=pd_spectral_data['Intensity']
        NormalizeSub_I=pd_spectral_data['Normalized_Sub_Intensity']
        NormalizeDi_I=pd_spectral_data['Normalized_Di_Intensity']
        NormalizeLn_I=pd_spectral_data['Normalized_Ln_Intensity']
        Nor_ax1=plt.subplot(427)
        Nor_ax1.plot(E_out_list,sum_result, marker='o', markersize=1, markerfacecolor='orchid',
                              markeredgecolor='orchid', linestyle='-', color='c', label='corrected peak spectra')
        Nor_ax1.set_xlabel('Energy(eV)',fontsize=12, color='#20B2AA')
        Nor_ax1.set_ylabel('intensity',fontsize=12, color='#20B2AA')
        Nor_ax1.set_title(f"Sum_Corrected Spectral_{filename}",loc='center')
        Nor_ax2=plt.subplot(428)
        Nor_ax2.plot(E_out_list,NormalizeSub_I, marker='o', markersize=1, markerfacecolor='orchid',
                              markeredgecolor='orchid', linestyle='-', color='c', label='corrected peak spectra')
        Nor_ax2.set_xlabel('Energy(eV)',fontsize=12, color='#20B2AA')
        Nor_ax2.set_ylabel('Sub_intensity',fontsize=12, color='#20B2AA')
        Nor_ax2.set_title(f"Normalize_Sub_Corrected Spectral_{filename}",loc='center')
        # save figure
        save_fig=os.path.join(self.save_path,f'Autocorrelated_img_{filename}.pdf')
        plt.savefig(save_fig)

if __name__=="__main__":
    root = Tk()
    root.withdraw()
    root.update()
    img_path = askopenfilename(title=u'Read CCD image')
    #bg_path = askopenfilename(title=u'Read background image')
    root.destroy()
    start_time=time.time()
    # bakground file
    bg_file=os.path.abspath("./CCD_TIF_Pipeline//tif_imgs//curve_line10_noise_background.tif")
    print(bg_file)
    # save corrected_list data
    save_folder,file=os.path.split(img_path)
    corr_folder=createPath(os.path.join(save_folder,'CorrectedResults'))
    filename,extension=os.path.splitext(file)
    print(f'save folder: {save_folder}\n filename:{filename}, type:{extension}')

    # process tif image
    TIF_Correction=TifAutoCorrelation(E_ref=450,E_ref_col=1200,dis_const=29.3)
    raw_matrix,file_title=TIF_Correction.input_tif_data(img_path)
    bg_matrix,bg_title=TIF_Correction.input_bg_data(bg_file)
    # process tif image
    p_col=1220
    half_n=100
    ROI_matrix=TIF_Correction.get_ROI_data(p_col=p_col,cut_lines=half_n)
    clearBG_matrix,median_matrix=TIF_Correction.tif_preprocess(ROI_matrix,False,False,True)
    # AutoCorrelation process
    min_result=TIF_Correction.minimal_FWHM_correlation(clearBG_matrix,slice_n=100,p_col=p_col)
    fit_paras=min_result[0]['fit_para']
    text_fit_paras=min_result[0]['para']
    min_p_col=min_result[-1]['p_col']
    min_slice_n=min_result[-1]['slice_n']

    print(f'get curve fit para with minimal FWHM={min_result[1]}\npara={fit_paras}\n{text_fit_paras}\n')
    print(f'minimal FWHM with slice num={min_slice_n} and center p_col={min_p_col}\n')
    # corrected image
    corr_peakdata=TIF_Correction.get_correct_peak_data(raw_matrix,fit_para=fit_paras,p_col=min_p_col)
    ROI_correct_data=TIF_Correction.get_correct_peak_data(ROI_matrix,fit_para=fit_paras,p_col=min_p_col)
    pd_spectrum_data=TIF_Correction.get_spectral_info(corr_peakdata,bg_data=bg_matrix,E_in=443.5,E_ref=450,p_col=min_p_col)

    print(f'full autocorrelation process cost {time.time()-start_time:.4f}s')
    # display corrected and spectral results
    TIF_Correction.plot_corrected_data(raw_matrix,ROI_matrix,corr_peakdata,ROI_correct_data,pd_spectrum_data,file_title)
    plt.show()

