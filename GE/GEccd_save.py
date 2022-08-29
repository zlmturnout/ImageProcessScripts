import numpy as np
import matplotlib.pyplot as plt
import tkinter,time
from tkinter import *
from tkinter.filedialog import askopenfilename
import matplotlib.image as mpimage
from PIL import Image
# coding: utf-8
import csv
from scipy import interpolate
import matplotlib.cm as cm

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
    print(type(temp))
    return u, v, temp


def xcorr(x, y, maxlags=10):
    Nx = len(x)
    if Nx != len(y):
        raise ValueError('x and y must be equal length')

    c = np.correlate(x, y, mode=2)

    if maxlags is None:
        maxlags = Nx - 1

    if maxlags >= Nx or maxlags < 1:
        raise ValueError('maxlags must be None or strictly positive < %d' % Nx)

    c = c[Nx - 1 - maxlags:Nx + maxlags]
    lag = np.linspace(-maxlags, maxlags, 2*maxlags+1)

    return c, lag


root = Tk()
root.withdraw()
root.update()
img_path = askopenfilename(title=u'Read CCD image')
#bg_path = askopenfilename(title=u'Read background image')
root.destroy()

t1 = time.time()
# img_path = r'D:\eline\REXS\code\data\spe2\Archive\align_1018\align17sept_1.tif'
# bg_path = r'D:\eline\REXS\code\data\20211215\-10degree_01_BGR_BGR.tif'

#matrix = mpimage.imread(img_path).astype('float64')
# add by limin
img = Image.open(img_path)
matrix = np.array(img,dtype=np.float32).T
#background = mpimage.imread(bg_path).astype('float64')
#matrix1 = matrix1.T


""" # Background subtraction
ExposureTime = 600 #seconds
background_aqn_time = 600 #seconds
extract_background=True
if extract_background:
    rawImageData = 1. * matrix
    if background.shape == matrix.shape[:2]:
        if background_aqn_time:
            matrix -= np.array(ExposureTime) * background \
                / background_aqn_time """
matrix1 = matrix.T
#matrix1 = matrix
plt.subplot(1,2,1),plt.imshow(matrix1,cmap=cm.rainbow,vmin=1300,vmax=1400)
plt.colorbar(location='bottom', fraction=0.05),plt.title("raw image")

thresholdUP = 0.9
thresholdDOWN = 0.1
matrix1 = detectorclean(matrix1, noise1=50, noise2=100)
print(type(matrix1),matrix1.shape)
m, n, out = clear_bg(matrix1)
matrix2 = out
new_img = np.zeros((m,n))
'''
plt.subplot(1,1,1)
plt.imshow(matrix)
plt.show()
'''


plt.subplot(1,2,2),plt.imshow(out,cmap=cm.rainbow,vmin=0,vmax=100),plt.title("clear background")
plt.colorbar(location='bottom', fraction=0.05)
plt.show()

j=6
index = 1100

k = 0.4 + j*0.1 +j**2*(1e-07)
low_lim = round(index*20 - 1200)
high_lim = round(index*20 + 1200)
#low_lim = 4000
#high_lim = 6400
new_X = np.arange(low_lim, high_lim)
result = np.zeros(len(new_X))
xinitial = np.arange(n)
xinterp = np.arange(0, n, 0.05)
xx = np.linspace(0, len(xinterp), n)
for ii in range(m):
    temp = matrix2[ii, :]
    ntemp = fastinterp1(xinitial, temp, xinterp)
    dd = round(k*ii)
    new_img[ii,:] = fastinterp1(new_X,ntemp[low_lim-dd:high_lim-dd],xx)
    #f=interpolate.interp1d(new_X,ntemp[low_lim-dd:high_lim-dd],kind='slinear')
    #new_img = f(xx)
    result = result + ntemp[low_lim-dd:high_lim-dd]
    
y_ = fastinterp1(new_X, result, xx)
fileout1 = xinitial[round(low_lim/20):round(high_lim/20)]
fileout2 = y_[round(low_lim/20):round(high_lim/20)]
fileout = np.array([fileout1, fileout2]).T
plt.plot(xinitial[round(low_lim/20):round(high_lim/20)],
            y_[round(low_lim/20):round(high_lim/20)], '.-')
plt.pause(0.5)

plt.show()

root = Tk()    # 创建一个Tkinter.Tk()
root.withdraw()      
fname = tkinter.filedialog.asksaveasfilename(title=u'保存文件', filetypes=[("csv", ".CSV")])
fname=fname
csvfile = open(f'{fname}.csv', 'w', newline="")  #打开方式还可以使用file对象
writer=csv.writer(csvfile)

#writer.writerows(data1)
writer.writerow(['pixels','intensity'])
data2 = fileout
writer.writerows(data2)
csvfile.close()
