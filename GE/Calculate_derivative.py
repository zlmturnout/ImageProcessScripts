# coding=utf-8
import os, sys, time
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
from scipy.misc import derivative
import pandas as pd
sys.path.append('.')

def import_data(filename):
    """
    import data from files as pandas dataframe,
    support filetype:<.xlsx>,<.csv>,<.json>.
    :return: data in pandas dataframe form
    """
    pd_data = pd.DataFrame()
    # filename, filetype = QFileDialog.getOpenFileName(self, "read data file(supported filetype:xlsx/csv/json)",
    #                                                  './', '*.xlsx;;*.csv;;*.json')
    # print(filename, filetype)
    assert os.path.isfile(os.path.abspath(filename))
    if filename.endswith('.xlsx'):
        # add dtype={'time stamp': 'datetime64[ns]'} if have 'time stamp'
        pd_data = pd.read_excel(filename, index_col=0, na_values=["NA"], engine='openpyxl')
        # print(pd_data)
    if filename.endswith('.csv'):
        pd_data = pd.read_csv(filename, index_col=0)
    if filename.endswith('.json'):
        pd_data = pd.read_json(filename)
    # drop the row with NaN and return
    return pd_data.dropna()


def cal_deriv(x:list, y:list,x_label:str='x',y_label:str='y'):  # x, y的类型均为列表
    diff_x = []  # 用来存储x列表中的两数之差
    for i, j in zip(x[0::], x[1::]):
        diff_x.append(j - i)

    diff_y = []  # 用来存储y列表中的两数之差
    for i, j in zip(y[0::], y[1::]):
        diff_y.append(j - i)

    slopes = []  # 用来存储斜率
    for i in range(len(diff_y)):
        slopes.append(diff_y[i] / diff_x[i])

    deriv = []  # 用来存储一阶导数
    for i, j in zip(slopes[0::], slopes[1::]):
        deriv.append((0.5 * (i + j)))  # 根据离散点导数的定义，计算并存储结果
    deriv.insert(0, slopes[0])  # (左)端点的导数即为与其最近点的斜率
    deriv.append(slopes[-1])  # (右)端点的导数即为与其最近点的斜率

    # plot results
    fig, ax = plt.subplots()
    axnew = ax.twinx()
    ax.set_title("direct derivative")
    lin0 = ax.plot(x, y, marker='o', markersize=3, markerfacecolor='#20B2AA',
                              markeredgecolor='#20B2AA', linestyle='-', color='#20B2AA', label='original')
    linD = axnew.plot(x, deriv,marker='o', markersize=3, linestyle='-', color='m', label='1st derivative')
    lins = lin0  + linD
    labels = [l.get_label() for l in lins]
    ax.legend(lins, labels, loc=0)
    axnew.grid(which='both')
    ax.set_ylabel(y_label, fontsize=16, color='#20B2AA')
    ax.set_xlabel(x_label, fontsize=16, color='#20B2AA')
    axnew.set_ylabel('1st derivative', fontsize=16, color='m')
    plt.show()
    return deriv,x


def interp_derivative(x: list, y: list, n_interp: int = 100,x_label:str='x',y_label:str='y'):
    """
    interpolate the x,y list first, then calculate the derivative value at each points
    :param n_interp: total number of the interpolate values
    :param x:
    :param y:
    :return:
    """
    interp_quad = interp1d(x, y, kind='quadratic')
    rbf=Rbf(x,y,function='linear')
    ius=InterpolatedUnivariateSpline(x,y) 
    n_interp=len(x)+n_interp
    new_x = [min(x) + i * (max(x) - min(x)) / n_interp for i in range(n_interp - 1)]
    new_y=ius(new_x)
    #new_y = interp_quad(new_x)
    deriv_val = list()
    for x_val in new_x[1::]:
        deriv_val.append(derivative(interp_quad, x_val, dx=1e-6))
    # deal with the new_x[0]
    slope0 = (new_y[1] - new_y[0]) / (new_x[1] - new_x[0])
    deriv_val.insert(0, slope0)

    # plot results
    fig,ax=plt.subplots()
    axnew=ax.twinx()
    ax.set_title("quadratic interpolate and derivative")
    lin0=ax.plot(x, y, 'or', markersize=3, label='original')
    lin1=ax.plot(new_x, new_y, '-xc', markersize=3, label='interpolate')
    linD=axnew.plot(new_x, deriv_val, '-om', markersize=3, label='1st derivative')
    lins=lin0+lin1+linD
    labels=[l.get_label() for l in lins]
    ax.legend(lins, labels, loc=0)
    axnew.grid(which='both')
    ax.set_ylabel(y_label, fontsize=16, color='#20B2AA')
    ax.set_xlabel(x_label, fontsize=16, color='#20B2AA')
    axnew.set_ylabel('1st derivative', fontsize=16, color='m')
    plt.show()
    return deriv_val,new_x

if __name__ == '__main__':
    xlsxFile1 = './GE/BPM_Zsize_0125_01_save.xlsx'
    xlsxFile2 = './2021-12-11/BeamSize_Z_pA_1211_M03_300um_save.xlsx'
    xlsxFile3 = './2021-12-11/BeamSize_Z_pA_1211_M04_300um_0.xlsx'
    xlsxFile4 = './2021-12-11/BeamSize_X_pA_1211_M02_1200um.xlsx'
    xlsxFile5 = './2021-12-11/BeamSize_X_pA_1211_M03_1200um_save.xlsx'
    xlsxFile6 = './2021-12-11/BeamSize_X_pA_1211_M04_1200um.xlsx'
    xlsxFile7 = './2021-12-11/BeamSize_X_pA_1211_M05_1200um.xlsx'

    print(os.path.abspath(xlsxFile1))
    pd_data=import_data(xlsxFile1)
    #print(pd_data)
    dict_data={}
    for label, content in pd_data.items():
        dict_data[str(label)] = content.tolist()
    print(dict_data['scan set'],dict_data['current(pA)'])
    x,y=dict_data['scan set'],dict_data['current(pA)']
    interp_derivative(x,y,len(x)+50,x_label='BPM-X(um)',y_label='Current(pA)')

    # method 2

    deriv_result = cal_deriv(x, y,x_label='BPM-X(um)',y_label='Current(pA)')
    plt.show()