import matplotlib.pyplot as plt
from numpy import exp, loadtxt, pi, sqrt
import os ,sys,time
from lmfit import Model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

def gaussian_fit(x, amplitude, mean, stddev):
    return amplitude * np.exp(-2*((x - mean) / stddev)**2)
#popt, _ = curve_fit(gaussian_fit, x, data)

def gaussian(x, amp, cen, wid):
    """1-d gaussian: gaussian(x, amp, cen, wid)"""
    return (amp / (sqrt(2*pi) * wid)) * exp(-(x-cen)**2 / (2*wid**2))

if __name__ == "__main__":
    datafile="./GE/corrected-BEST_12-C@445eV0907.xlsx"
    datafile2='./GE/04_Z_pA_BPM_Z_slit2-50_Size_SM6_pitch-1dot07.xlsx'
    print(os.path.abspath(datafile))
    pd_data=import_data(datafile)
    x = pd_data.values[:, 0]
    y = pd_data.values[:, 3]
    gmodel = Model(gaussian)
    result = gmodel.fit(y, x=x, amp=1, cen=1200, wid=1)
    print(result.values)
    print(result.fit_report())
    wid_fit=result.params['wid'].value
    wid_err=result.params['wid'].stderr
    FWHM=wid_fit*2*sqrt(np.log(4))
    FWHM_err=wid_err*2*sqrt(np.log(4))
    print(f'get FWHM={FWHM} with error +/-{FWHM_err}')
    plt.plot(x, y, 'o')
    plt.plot(x, result.init_fit, '--', label='initial fit')
    plt.plot(x, result.best_fit, '-', label='best fit')
    plt.legend()
    plt.show()