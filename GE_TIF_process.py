import time, random, sys, os, math
from PIL import Image
import pandas as pd
import numpy as np
from PySide6.QtWidgets import QFileDialog
import tkinter as tk
from tkinter import filedialog

# for tinker interface
root = tk.Tk()
root.withdraw()

def open_tif_img(img_path:str=None):
    """
    open a 16bit tif file and return filename,img_data
    :return: filename,img_data in np array
    """
    img_data = np.array([])
    if  not img_path or not os.path.isdir(img_path):
        img_path='/'
    filename = filedialog.askopenfilename(initialdir = img_path,title = "Select tif file",filetypes = (("jpeg files","*.jpg"),("tif file","*.tif")))
    print(f'get file: {filename}')
    foldername, filetype=os.path.splitext(filename)
    print(foldername, filetype)
    if os.path.isfile(filename) and filename.endswith('.tif'):
        img = Image.open(filename)
        img_data = np.array(img)
        print(f'shape of the read img={np.shape(img_data)}')
    return filename, img_data

if __name__ == "__main__":
    #download_path = r'F:\BeautifulPictures'
    download_path=r'E:\迅雷下载\MaryMoody'
    save_path = r'F:\Beautyleg'
    img_path=r'E:\areaDetector\saved data'
    tif_file, img_data=open_tif_img(img_path)
    print(f'get tif pic:{tif_file} with image date {img_data}')