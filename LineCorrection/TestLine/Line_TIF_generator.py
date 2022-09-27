import os, sys,time,datetime,random,math
from numpy import exp, loadtxt, pi, sqrt,log
import numpy as np
from PIL import Image
import csv,cv2
import pandas as pd
sys.path.append('.')

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


if __name__ == '__main__':
    # tif_file = r"F:\\Eline20U2\\ElineData\\DATA2022\\20220905\\01-backup.tif"
    # img = Image.open(tif_file)
    # matrix = np.array(img,dtype=np.float32)
    # print(img.info)
    width=30
    add_noise=True
    if add_noise:
        noise_str='noise'
    else:
        noise_str='clearBG'
    new_tif=os.path.join(os.path.join('./GE/tif_files'),f'new_testline{width}_{noise_str}.tif')
    #pil_image=Image.fromarray(matrix)
    #pil_image.show()
    #tiffinfo={'compression': 'raw', 'dpi': (1, 1), 'resolution': (1, 1)}
    #pil_image.save(new_tif)
    time_start=time.time()
    img_matrix=generate_Linetif(width=width,tif_shape=(2048,2052),add_noise=add_noise)
    pil_image=Image.fromarray(img_matrix)
    pil_image.save(new_tif)
    print(f'line tif generate cost: {time.time()-time_start:.4f}s')