# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 12:39:01 2023

@author: MaYutong
"""

import netCDF4 as nc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


# %%


def read_lcc():
    inpath = (r"E:/ESA/ESA_LUC_2000_SPEI03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        # print(f.variables['lccs_class'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        lcc = (f.variables['lcc'][:])

    return lcc


# %%
def read_nc():
    global lat, lon
    inpath = (r"E:/LAI4g/data_MG/LAI_Detrend_Zscore_82_20_MG_SPEI0.5x0.5.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        # print(f.variables['lai'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        lai = (f.variables['lai'][:, 4:9, :, :])
        lai_gsl = lai.reshape(39*5, 30, 50)

    return lai_gsl


# %%


def read_nc2():
    inpath = (rf"E:/SPEI_base/data/spei03_MG_detrend.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        spei = (f.variables['spei'][:, 1:, ])
        
    spei_gsl = spei[4:9, :].reshape((39*5, 30, 50), order='Fa')
    return spei_gsl


# %% 滑动相关计算


def roll_corr(data1, data2, windows):
    corr = np.zeros((195, 30, 50))
    for i in range(30):
        for j in range(50):
            a = pd.Series(data1[:, i, j], name="LAI")
            b = pd.Series(data2[:, i, j], name="SPEI")
            if np.isnan(a).any() or np.isnan(b).any():
                corr[:, i, j] = np.nan
            else:
                df = pd.concat([a, b], axis=1)
                r = df.rolling(windows).corr().iloc[1::2, 0]
                corr[:, i, j] = r

    return corr

# %%生成新的nc文件
def CreatNC(windows, data1):
    new_NC = nc.Dataset(
        rf"E:/LAI4g/data_MG/moving corr/LAI_SPEI_detrend_Win{windows}.nc", 'w', format='NETCDF4')

    time = np.arange(195)

    new_NC.createDimension('time', 195)
    new_NC.createDimension('lat', 30)
    new_NC.createDimension('lon', 50)

    var = new_NC.createVariable('corr', 'f', ("time", "lat", "lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))

    new_NC.variables['corr'][:] = data1
    new_NC.variables['lat'][:] = lat
    new_NC.variables['lon'][:] = lon

    # 最后记得关闭文件
    new_NC.close()


# %% 数据读取
lcc = read_lcc()
lai = read_nc()
spei = read_nc2()


for win in [10, 20, 30, 40, 50][:]:
    r = roll_corr(lai, spei, win)
    CreatNC(win, r)

