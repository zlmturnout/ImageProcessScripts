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

def fastinterp1(x, y, xi):
    ixi = np.digitize(xi, x)
    n = len(x)
    ixi[ixi == n] = n - 1 
    t = (xi - x[ixi-1])/(x[ixi] - x[ixi-1])
    yi = (1-t) * y[ixi-1] + t * y[ixi]
    yi = yi.T
    return yi

if __name__ == "__main__":
    num=9
    x=np.arange(num)
    # random number
    y=np.random.randn(num)*10
    xinterp = np.arange(0, num, 0.05)
    print(len(y))
    y_interp=fastinterp1(x,y,xinterp)
    print(len(y_interp))
    plt.plot(x,y,marker='o', markersize=4)
    plt.plot(xinterp,y_interp,marker='x', markersize=1)
    plt.legend(["initial","fast interpolate"])
    plt.show()