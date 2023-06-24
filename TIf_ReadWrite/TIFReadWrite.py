import time, random, sys, os, math
from PIL import Image
from scipy import misc
from PIL import Image
import pandas as pd
import numpy as np
from PySide6.QtWidgets import QFileDialog
import tkinter as tk
from tkinter import filedialog
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import cv2
from tifffile import TiffFile,tifffile

def Read_Tif_XMP(tif_file:str):
    """read tif image by tifffile

    Args:
        tif_file (str): _description_
    Returns:
        tif_data (np.array),img_Description(list(dicts))
    """
    tif_data=np.array([])
    if os.path.isfile(tif_file) and tif_file.endswith('.tif'):
        img = Image.open(tif_filepath)
        tif_data = np.array(img,dtype=np.float32)
        xmp_all=img.getxmp()['xmpmeta']
        xmptk=xmp_all['xmptk']
        print(f'acquire equipment name :{xmptk}')
        xmp_RDF=xmp_all['RDF']
        img_Description_dictlist=xmp_all['RDF']['Description']
    return tif_data,img_Description_dictlist

if __name__ == "__main__":
    tif_filepath=os.path.abspath('./Tif_ReadWrite/R1_11-C@445eV_600s.tif')
    img = Image.open(tif_filepath)
    img_data = np.array(img,dtype=np.float32)
    print(tif_filepath)
    tif=TiffFile(tif_filepath)
    xmp=tif.pages[0].tags["XMP"].value
    print(f'XMP info:{xmp.decode()}')
    print("get tif info")
    tif_data,tif_desc=Read_Tif_XMP(tif_filepath)
    for desc in tif_desc:
        for key,value in desc.items():
            print(f'{key}:{value}')