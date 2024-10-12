import time, random, sys, os, math
from PIL import Image
from scipy import misc
from PIL import Image
import pandas as pd
import matplotlib
import matplotlib.cm as colormap
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
#matplotlib.use("Agg")
from matplotlib.animation import FFMpegWriter
import cv2
import numpy as np
from functools import partial
from tifffile import TiffFile,tifffile



def get_I_range(tif_data:np.array):
    """get the range of pixel intensity of the image

    Args:
        tif_data (np.array): img array
    Returns:
        (I_min:int,I_max:int):min, aver, median, max, percentile80 intensity of the img array
    """
    w, h = np.shape(tif_data)
    if w > 0 and h > 0:
        return (int(np.min(tif_data)),int(np.average(tif_data)),
                int(np.median(tif_data)),int(np.max(tif_data)),int(np.percentile(tif_data,80)))

def load_txt_to_dict(txt:str,split_symbol:str=':'):
    with open(txt,'r') as f:
        data=f.readlines()
    data_dict={}
    for line in data:
        line=line.strip('\n')
        line=line.split(split_symbol)
        data_dict[line[0]]=line[1]
    return data_dict

def load_tif_img(filename:str):
    with Image.open(filename) as img:
        img_data = np.array(img,dtype=np.float32)
        print(f'shape of the read img={np.shape(img_data)}')
        # save raw image info
    return img_data

def construct_tif_params(tif_folder:str,key_name:str="DET_2theta"):
    """construct a dict={"key_name":[value_list],"file_name":[file_list],"file_name":tif_folder} 
         value_list->file_list: each value related to one file

    Args:
        tif_folder (str): _description_
        key_name (str, optional): _description_. Defaults to "DET_2theta".

    Returns:
        _type_: _description_
    """
    tif_para_dict={"value_list":[],"file_list":[],"file_folder":tif_folder,"key_name":key_name} 
    if not os.path.exists(tif_folder):
        print(f'{tif_folder} does not exist')
        return tif_para_dict
    for file in os.listdir(tif_folder):
        if file.endswith('tif'):
            tif_file=os.path.join(tif_folder,file)
            txt_file=os.path.join(tif_folder,file.replace('tif','txt'))
            if os.path.isfile(os.path.join(tif_folder,txt_file)):
                # the txt file is inside with the same filename_noext
                data_dict=load_txt_to_dict(txt_file)
                value=data_dict.get(key_name,-1) # -1 if key_name not in data_dict
                tif_para_dict["value_list"].append(value)
                tif_para_dict["file_list"].append(file)
    # sort the dic
    new_tif_para_dict={"value_list":[],"file_list":[],"file_folder":tif_folder,"key_name":key_name}
    sort_id=sorted(range(len(tif_para_dict["value_list"])),key=lambda k:tif_para_dict["value_list"][k],reverse=True)
    new_tif_para_dict["value_list"]=[tif_para_dict["value_list"][k] for k in sort_id]
    new_tif_para_dict["file_list"]=[tif_para_dict["file_list"][k] for k in sort_id]
    return new_tif_para_dict

def plot_tif_img_with_params(filename:str,params:dict,ax:plt.Axes):
    """plot the tif img with parameters in title
    params={para_name:value}

    Args:
        filename (str): tif file
        params (dict): {para_name:value}
        ax (plt.Axes): axes for plot

    Returns:
        _type_: _description_
    """
    with Image.open(filename) as img:
        img_data = np.array(img,dtype=np.float32)
        I_min,I_aver,I_median,I_max,I_80per=get_I_range(img_data)
        I_80per+=I_median-I_min # vmin=1500,vmax=3600
    im = ax.imshow(img_data, cmap=colormap.rainbow,vmin=I_min,vmax=I_80per)
    #ax.set_title(f"I_min:{I_min:.2f},I_aver:{I_aver:.2f},I_median:{I_median:.2f}")
    for key,value in params.items():
        #text=ax.text(x=100,y=125,s=f"{key}:{value}",color='white',fontsize=20,va='bottom',ha='left')
        text=ax.set_title(f"{key}:{value}",color='red',loc='center')
    return im,text

def ini_tif_fig(ax:plt.Axes):
    ax.cla()
    return [ax]

def update_tif_fig(frame,tif_param_dict:dict,ax:plt.Axes):
    param_dict={tif_param_dict.get("key_name","unknown_Param"):tif_param_dict["value_list"][frame]}
    tif_folder=tif_param_dict.get("file_folder")
    tif_file=os.path.join(tif_folder,tif_param_dict["file_list"][frame])
    im=plot_tif_img_with_params(tif_file,param_dict,ax)
    return [ax]

    
if __name__=='__main__':
    fig, ax = plt.subplots()
    #Tif_folder=os.path.abspath(r'Tif_ReadWrite/STO-2theta')
    Tif_folder=os.path.abspath(r'L:\REXS_CCD1024\2023-10-16\STO-1795-1805eV')
    #tif_file_list
    tif_para_dict=construct_tif_params(Tif_folder,key_name="energy(eV)") #energy(eV) DET_2theta
    print(tif_para_dict)
    # single plot
    # tif_0=os.path.join(Tif_folder,tif_para_dict["file_list"][0])
    # tif_para_0={tif_para_dict['key_name']:tif_para_dict['value_list'][0]}
    # plot_tif_img_with_params(tif_0,tif_para_0,ax)
    tif_num=len(tif_para_dict["file_list"])
    print(f' total frame:{tif_num}')
    # animation
    
    # ani = FuncAnimation(fig, partial(update_tif_fig,tif_param_dict=tif_para_dict,ax=ax), frames=range(tif_num),
    #                     init_func=partial(ini_tif_fig,ax),blit=False,interval=300,repeat=True)
    # plt.show()
    # save tp mp4
    moviewriter = FFMpegWriter()
    mp4_file=os.path.join(Tif_folder,"tif_change_200dpi.mp4")
    with moviewriter.saving(fig, mp4_file, dpi=200):
        for frame in range(tif_num):
            update_tif_fig(frame,tif_param_dict=tif_para_dict,ax=ax)
            moviewriter.grab_frame()

