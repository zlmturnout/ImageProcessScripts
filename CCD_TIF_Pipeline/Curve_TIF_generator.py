import os, sys,time,datetime,random,math
from numpy import exp, loadtxt, pi, sqrt,log
import numpy as np
from PIL import Image
import csv,cv2
import pandas as pd
sys.path.append('.')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

def gaussian(x, amp, cen, wid):
    """1-d gaussian: gaussian(x, amp, cen, wid)"""
    return (amp / (sqrt(2*pi) * wid)) * exp(-(x-cen)**2 / (2*wid**2))


def generate_Linetif(width:int,tif_shape:tuple=(2048,2052),add_noise=True):
    """generate a np array containing line with width of several pixels 
    line_expression:y=57*x-57000
    Args:
        width (int): witdth of the line in pixels
    """
    img_matrix=np.zeros(shape=tif_shape)
    w=0.5*width/sqrt(log(4)) # gaussian 2w=FWHM/sqrt(ln4)
    for row in range(tif_shape[1]): # row is height=2052
        for col in range(tif_shape[0]): # row is width=2048
            # distance of point(j,row) to the line
            dist=abs((col-(57000+row)/57)*math.sin(np.arctan(57)))
            noise_num=0
            if add_noise:
                noise_num=100*random.random()
            img_matrix[col,row]=float(1300+1000*gaussian(dist,1,0,w)+0.005*col+15+noise_num)
            # if round(dist)<width:
            #     img_matrix[col,row]=float(1300+1000*gaussian(dist,1,0,w))
            # else:
            #     img_matrix[col,row]=float(1200+10*random.random())
    #print(img_matrix)
    return img_matrix.T

def generate_2st_curve_tif(width:int,y0:float,y1:float,y2:float,tif_shape:tuple=(2048,2052)
                           ,add_noise=True,linear_noise=True,liner_aspect:float=0.05,only_bg=False):
    """generate a np array containing secondary curve line with width of several pixels 
    
    line_expression:col-X=y0+y1*Y+y2*Y**2

    Args:
        width (int): _description_
        x0 (float): _description_
        x1 (float): _description_
        x2 (float): _description_
        tif_shape (tuple, optional): _description_. Defaults to (2048,2052).
        add_noise (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    img_matrix=np.zeros(shape=tif_shape)
    w=0.5*width/sqrt(log(4)) # gaussian 2w=FWHM/sqrt(ln4)
    for row in range(tif_shape[1]): # row is height=2052
        for col in range(tif_shape[0]): # column is width=2048
            # distance of point(col,row) to the point (x,x0+x1*x+x2*x**2) at the line 
            #dist=abs((col-(57000+row)/57)*math.sin(np.arctan(57)))
            dist=abs(col-(y0+y1*row+y2*row**2))
            noise_num=0
            liner_a=0
            counts=1000
            if add_noise:
                noise_num=100*random.random()
            if linear_noise:
                liner_a=liner_aspect
            if only_bg:
                counts=0
            img_matrix[col,row]=float(1300+counts*gaussian(dist,1,0,w)+liner_a*col+15+noise_num)
            #img_matrix[col,row]=float(1300+1000*gaussian(dist,1,0,w)+0.000*col+15)
    return img_matrix.T



if __name__ == '__main__':
    # tif_file = r"F:\\Eline20U2\\ElineData\\DATA2022\\20220905\\01-backup.tif"
    # img = Image.open(tif_file)
    # matrix = np.array(img,dtype=np.float32)
    # print(img.info)
    width=15
    add_noise=True
    only_bg=False
    if add_noise:
        noise_str='noise'
    else:
        noise_str='clearBG'
    if only_bg:
        bg_str='background'
    else:
        bg_str='gaussian'
    new_tif=os.path.join(os.path.join('./CCD_TIF_Pipeline/tif_imgs'),f'curve_line{width}_{noise_str}_{bg_str}.tif')
    #pil_image=Image.fromarray(matrix)
    #pil_image.show()
    #tiffinfo={'compression': 'raw', 'dpi': (1, 1), 'resolution': (1, 1)}
    #pil_image.save(new_tif)
    time_start=time.time()
    #y=y=x0+x1*x+x2*x**2
    y0,y1,y2=1201,1.72e-02,8.77e-08
    img_matrix=generate_2st_curve_tif(width=width,y0=y0,y1=y1,y2=y2,tif_shape=(2048,2052),
                add_noise=add_noise,linear_noise=True,liner_aspect=0.02,only_bg=only_bg)
    plt.subplot(1,2,1),plt.imshow(img_matrix,cmap=cm.rainbow,vmin=1300,vmax=1400)
    plt.colorbar(location='bottom', fraction=0.1),plt.title("generate curve image")
    # select one row to plot
    #print(img_matrix[1])
    select_array=img_matrix[1:200]
    row,col=np.shape(select_array)
    print(f'select array with row={row},column={col}')
    average_array=np.sum(select_array,axis=0)/row
    plt.subplot(1,2,2),plt.plot(average_array)
    pil_image=Image.fromarray(img_matrix)
    pil_image.save(new_tif)
    print(f'line tif generate cost: {time.time()-time_start:.4f}s')
    plt.show()