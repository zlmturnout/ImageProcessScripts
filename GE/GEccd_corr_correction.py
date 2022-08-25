import numpy as np
import matplotlib.pyplot as plt
import tkinter
from tkinter import *
from tkinter.filedialog import askopenfilename
import matplotlib.image as mpimage
import csv


info = {}
info['element'] = 'TiO2'
info['energy'] = 458

def fastinterp1(x, y, xi):
    ixi = np.digitize(xi, x)
    n = len(x)
    ixi[ixi == n] = n - 1
    t = (xi - x[ixi-1])/(x[ixi] - x[ixi-1])
    yi = (1-t) *y[ixi-1] + t * y[ixi]
    yi = yi.T
    return yi

def detectorclean(exp, noise1, noise2):
    
    exp = exp - np.mean(exp[:, noise1:noise2])
    exp[exp>(np.max(exp) * thresholdUP)] = 0
    exp[exp<(np.min(exp) * thresholdDOWN)] = 0
    detectorcleanout = exp
    return detectorcleanout


def clear_bg(exp):
    u, v = exp.shape
    temp = np.zeros((u,v))
    out = np.zeros((u,v))
    
    for i in np.arange(u):
        k = (np.sum(exp[i,1:10]) - np.sum(exp[i,-10:-1]))/(v)/10
        b = np.sum(exp[i,1:10])/10 - k*10
        exp_bg = -k * np.arange(v) + b 
        temp[i,:] = exp[i,:] - exp_bg
    #print(type(temp))
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

#img_path = r'D:\eline\REXS\code\data\spe2\Archive\align_1018\align17sept_1.tif'
#bg_path = r'D:\eline\REXS\code\data\20211215\-10degree_01_BGR_BGR.tif'

matrix= mpimage.imread(img_path).astype('float64')
#background = mpimage.imread(bg_path).astype('float64')
print(matrix.shape)

'''
# Background subtraction
ExposureTime = 600 #seconds
background_aqn_time = 600 #seconds
extract_background=True
if extract_background:
    rawImageData = 1. * matrix
    if background.shape == matrix.shape[:2]:
        if background_aqn_time:
            matrix -= np.array(ExposureTime) * background \
                / background_aqn_time
'''

#show raw CCD image
plt.subplot(2, 2,1)
plt.imshow(matrix.astype('uint8'))
plt.title("raw image")
plt.colorbar()

matrix1 = matrix
thresholdUP=0.9
thresholdDOWN = 0.1
matrix1 = detectorclean(matrix1, noise1 = 1, noise2 = 100)
#print(type(matrix1))
m, n, out = clear_bg(matrix1)
matrix = out
#matrix = out.astype('uint8')
#plt.subplot(2,1,2)
#plt.imshow(matrix)
#plt.show()
#cv.imwrite(r'D:\eline\RXES\code\data\RXES1227\shift\TiO2_458eV_SHIFT30_1h_slit2_50.tif', matrix)

matrix_org = matrix
expsum_org = np.sum(matrix_org, axis = 0)

'''
plt.subplot(2,2,3)
plt.plot(expsum_org)
plt.xlim([900, 1100])
'''

steps =0.25
#m, n = matrix.size
xinitial = np.arange(n)
xinterp = np.arange(0,n-1+steps,steps)
largeexp = np.zeros((m,int((n-1)/steps+1)))

shift = np.zeros((m,1))
xshift = np.arange(m)
xshift1 = xshift.reshape(-1,1)
pixelini = 1
pixelfin1 = 2000
pixelfin2 = 2000
###deviations
maxshift = 25
dev = round(1.5 * maxshift)
#lag = []
corrfig = []
location = []
for cur in range(m):
    largeexp[cur,:] = fastinterp1(xinitial, matrix[cur,:], xinterp)


reflin = round(m/2)
for cur in range(m):
    corrfig, lag = xcorr(largeexp[reflin,:],largeexp[cur,:], maxlags=round(dev*0.5/steps))
    location = np.argmax(corrfig)
    shift[cur] = lag[location]
shift[shift>(maxshift/steps)] = 0
shift[shift<(-maxshift/steps)] = 0

p = np.polyfit(xshift[pixelini:pixelfin1], shift[pixelini:pixelfin2], 2)
linshift = np.polyval(p, xshift1)#多项式校正后的值
#print("Quadratic Coefficient = %f;\n Coefficient of first order = %f;\n \
#     Constant = %f" % (p[0], p[1],p[2]))
print(p)
#plt.plot(xshift, shift*steps)
#plt.plot(xshift, linshift*steps)
linshift = np.round(linshift)
largeexpt = largeexp.T


#integrate
for cur in range(m):
    if linshift[cur] >0:
        largeexpt[:, cur] = np.roll(largeexpt[:,cur],int(linshift[cur]))
        kk = int(linshift[cur])
        for i in range(kk):
            largeexpt[i,cur] = 0
    elif linshift[cur] <0:
        largeexpt[:, cur] = np.roll(largeexpt[:,cur],int(linshift[cur]))
        u1 = int(np.size(xinterp) - (-linshift[cur]) + 1)
        u2 = np.size(xinterp)
        for i in range(u1, u2):
            largeexpt[i,cur] = 0

largeexp = largeexpt.T

for cur in range(m):
    matrix[cur,:] = fastinterp1(xinterp, largeexp[cur,:], xinitial)

# processed image
plt.subplot(2,2,2)
plt.imshow(matrix)
plt.colorbar()
plt.title("processed image")


expsum = np.sum(matrix, axis = 0)
matrixsum = expsum
#show spectrum
plt.subplot(2,2,3)
plt.plot(expsum_org)
plt.plot(matrixsum)
plt.legend(['uncorrected','corrected'], loc = 'upper right')
#plt.xlim([200, 440])
plt.show()


fileout1 = xinitial
fileout2 = matrixsum
fileout = np.array([fileout1, fileout2]).T

root = Tk()    # 创建一个Tkinter.Tk()实例
root.withdraw()      
fname = tkinter.filedialog.asksaveasfilename(title=u'保存文件', filetypes=[("csv", ".CSV")])
fname=fname
csvfile = open(f'{fname}.csv', 'w', newline="")  #打开方式还可以使用file对象
writer=csv.writer(csvfile)
#writer.writerow(['元素','能量'])#写入一行
#data1=[( '元素','能量'),(info['element'],info['energy']), \
    #('二次项系数','一次项系数','常数' ), (p[0],p[1],p[2])]

#writer.writerows(data1)
writer.writerow(['pixels','intensity'])
data2 = fileout
writer.writerows(data2)
csvfile.close()