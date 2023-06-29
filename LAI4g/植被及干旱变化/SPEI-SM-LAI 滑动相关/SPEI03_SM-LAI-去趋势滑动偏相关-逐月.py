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

plt.rcParams['font.sans-serif']=['simsun']
plt.rcParams['axes.unicode_minus'] = False

import warnings
warnings.filterwarnings("ignore")

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

    spei_gsl = spei[4:9, :].reshape((39*5, 30, 50), order='F')
    return spei_gsl


# %%

def MG(data):
    t = np.arange(1, 481, 1)
    l = np.arange(1, 5, 1)
    spei_global = xr.DataArray(data, dims=['t', 'l', 'y', 'x'], coords=[
                               t, l, lat2, lon2])  # 原SPEI-base数据
    spei_rg1 = spei_global.loc[:, :, 40:55, 100:125]
    spei_rg1 = np.array(spei_rg1)
    return spei_rg1


def read_nc3():
    global lat2, lon2
    # inpath = rf"E:/GLDAS Noah/DATA_RG/SM_81_20_ORINGINAL_SPEI0.5x0.5.nc"
    inpath = rf"E:/GLDAS Noah/DATA_RG/SM_Zscore_81_20_SPEI0.5x0.5.nc"
    with nc.Dataset(inpath, mode='r') as f:
        # print(f.variables.keys())
        lat2 = (f.variables['lat'][:])
        lon2 = (f.variables['lon'][:])
        sm = (f.variables['sm'][:, :, :])
        sm_MG = MG(sm)
        sm = sm_MG.reshape(40, 12, 4, 30, 50)
        sm_gsl = sm[:, 4:9, :]
        sm_gsl = sm_gsl.reshape(200, 4, 30, 50)
    return sm_gsl[5:, 0, ]

# %% 滑动相关计算


def roll_corr(data1, data2, data3, windows):
    pcorr1 = np.zeros((195, 30, 50))
    pcorr2 = np.zeros((195, 30, 50))
    for i in range(30):
        for j in range(50):
            a = pd.Series(data1[:, i, j], name="LAI")
            b = pd.Series(data2[:, i, j], name="SPEI")
            c = pd.Series(data3[:, i, j], name="SM")
            if np.isnan(a).any() or np.isnan(b).any() or np.isnan(c).any():
                pcorr1[:, i, j] = np.nan
                pcorr2[:, i, j] = np.nan
            else:
                df = pd.concat([a, b, c], axis=1)
                for k, df_win in enumerate(df.rolling(windows)):
                    # print(df_win)
                    if k+1 >= windows:
                        corr = df_win.pcorr()
                        pcorr1[k, i, j] = corr.iloc[0, 1]
                        pcorr2[k, i, j] = corr.iloc[0, 2]
                    else:
                        pcorr1[k, i, j] = np.nan
                        pcorr2[k, i, j] = np.nan

    return pcorr1, pcorr2

# %%生成新的nc文件


def CreatNC(windows, data1, data2):
    new_NC = nc.Dataset(
        rf"E:/LAI4g/data_MG/moving corr/LAI_SPEI_SM_Partial_detrend_Win{windows}.nc", 'w', format='NETCDF4')

    time = np.arange(195)

    new_NC.createDimension('time', 195)
    new_NC.createDimension('lat', 30)
    new_NC.createDimension('lon', 50)

    var = new_NC.createVariable('corr1', 'f', ("time", "lat", "lon"))
    new_NC.createVariable('corr2', 'f', ("time", "lat", "lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))

    new_NC.variables['corr1'][:] = data1
    new_NC.variables['corr2'][:] = data2
    new_NC.variables['lat'][:] = lat
    new_NC.variables['lon'][:] = lon

    var.description = f"滑动相关的滑动窗口为{windows}，corr1为LAI和SPEI的偏相关（去除表层SM），corr2为LAI和SM的偏相关（去除SPEI）"

    # 最后记得关闭文件
    new_NC.close()


# %% 数据读取
# lcc = read_lcc()
lai = read_nc()
spei = read_nc2()
sm1 = read_nc3()

for win in [10, 20, 30, 40, 50][:]:
    r1, r2 = roll_corr(lai, spei, sm1, win)
    CreatNC(win, r1, r2)
    print(f"Windows {win} is done!")
