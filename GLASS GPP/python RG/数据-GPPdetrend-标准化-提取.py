# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 20:32:28 2022

@author: MaYutong
"""
import datetime as dt
import netCDF4 as nc
from matplotlib import cm
from matplotlib import font_manager
import matplotlib.dates as mdate
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
from scipy import signal
from sklearn import preprocessing
import xarray as xr


# %%
def read_nc(inpath):
    global lat2, lon2
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())

        gpp = (f.variables['GPP'][:])
        lat2 = (f.variables['lat'][:])
        lon2 = (f.variables['lon'][:])

    return gpp


def sif_xarray2(band1):
    mn = np.arange(12)
    sif = xr.DataArray(band1, dims=['mn', 'y', 'x'], coords=[mn, lat2, lon2])

    sif_MG = sif.loc[:, 40:55, 100:125]
    return np.array(sif_MG)


def exact_data1():
    for yr in range(1982, 2019):
        inpath = rf"E:/GLASS-GPP/Month RG/GLASS_GPP_RG_SPEI_0.5X0.5_{yr}.nc"
        data = read_nc(inpath)

        data_MG = sif_xarray2(data)
        data_MG = data_MG.reshape(1, 12, 30, 50)

        if yr == 1982:
            data_all = data_MG
        else:
            data_all = np.vstack((data_all, data_MG))

    return data_all


# %% 去趋势和标准化
def detrend(data):
    data_detrend = np.array(data)
    for mn in range(12):
        for r in range(30):
            for c in range(50):
                test = data_detrend[:, mn, r, c]
                contain_nan = (True in np.isnan(test))
                if contain_nan == True:
                    data_detrend[:, mn, r, c] = np.nan
                else:
                    detrend = signal.detrend(test, type='linear')
                    data_detrend[:, mn, r, c] = detrend+test.mean()

    return data_detrend


def Zscore(data):
    data_z = np.array(data)
    for mn in range(12):
        for r in range(30):
            for c in range(50):
                test = data_z[:, mn, r, c]
                contain_nan = (True in np.isnan(test))
                if contain_nan == True:
                    data_z[:, mn, r, c] = np.nan
                else:
                    data_z[:, mn, r, c] = preprocessing.scale(
                        data[:, mn, r, c])

    return data_z

# %% 去趋势


def trend(data):
    t = np.arange(1, 35, 1)
    s, r0, p = np.zeros((300, 600)), np.zeros(
        (300, 600)), np.zeros((300, 600))

    for r in range(300):
        if r % 30 == 0:
            print(f"{r} is done!")
        for c in range(600):
            a = data[:, r, c]
            if np.isnan(a).any():
                s[r, c], r0[r, c], p[r, c] = np.nan, np.nan, np.nan

            else:
                s[r, c], _, r0[r, c], p[r, c], _ = linregress(t, a)

    return s, p
# %%


def CreatNC(data1, data2):
    month = np.arange(12)
    yr = np.arange(1982, 2019)
    lat_MG = np.arange(40.25, 55, 0.5)
    lon_MG = np.arange(100.25, 125, 0.5)

    new_NC = nc.Dataset(
        rf"E:/GLASS-GPP/MG detrend Zscore/MG_GPP_Detrend_Zscore_SPEI_0.5X0.5.nc",
        'w', format='NETCDF4')

    new_NC.createDimension('yr', 37)
    new_NC.createDimension('month', 12)
    new_NC.createDimension('lat', 30)
    new_NC.createDimension('lon', 50)

    var = new_NC.createVariable('GPP_de', 'f', ("yr", "month", "lat", "lon"))
    new_NC.createVariable('GPP_Z', 'f', ("yr", "month", "lat", "lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))

    new_NC.variables['GPP_de'][:] = data1
    new_NC.variables['GPP_Z'][:] = data2
    new_NC.variables['lat'][:] = lat_MG
    new_NC.variables['lon'][:] = lon_MG

    var.description = "Units: gC m-2 month-1; FillValue: 65535, 已处理为np.nan, scale也已处理过"

    # 最后记得关闭文件
    new_NC.close()

# %%


def CreatNC2(data1):
    month = np.arange(12)
    yr = np.arange(1982, 2019)
    lat_MG = np.arange(40.25, 55, 0.5)
    lon_MG = np.arange(100.25, 125, 0.5)

    new_NC = nc.Dataset(
        rf"E:/GLASS-GPP/MG detrend Zscore/MG_GPP_Zscore_SPEI_0.5X0.5.nc",
        'w', format='NETCDF4')

    new_NC.createDimension('yr', 37)
    new_NC.createDimension('month', 12)
    new_NC.createDimension('lat', 30)
    new_NC.createDimension('lon', 50)

    var = new_NC.createVariable('GPP_Z', 'f', ("yr", "month", "lat", "lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))

    new_NC.variables['GPP_Z'][:] = data1
    new_NC.variables['lat'][:] = lat_MG
    new_NC.variables['lon'][:] = lon_MG

    var.description = "Units: gC m-2 month-1; FillValue: 65535, 已处理为np.nan, scale也已处理过"

    # 最后记得关闭文件
    new_NC.close()

# %%


def CreatNC3(data1):
    month = np.arange(12)
    yr = np.arange(1982, 2019)
    lat_MG = np.arange(40.25, 55, 0.5)
    lon_MG = np.arange(100.25, 125, 0.5)

    new_NC = nc.Dataset(
        rf"E:/GLASS-GPP/Month Global SPEI0.5x0.5/MG_GPP_82_18_SPEI_0.5X0.5.nc",
        'w', format='NETCDF4')

    new_NC.createDimension('yr', 37)
    new_NC.createDimension('month', 12)
    new_NC.createDimension('lat', 30)
    new_NC.createDimension('lon', 50)

    var = new_NC.createVariable('GPP', 'f', ("yr", "month", "lat", "lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))

    new_NC.variables['GPP'][:] = data1
    new_NC.variables['lat'][:] = lat_MG
    new_NC.variables['lon'][:] = lon_MG

    var.description = "Units: gC m-2 month-1; FillValue: 65535, 已处理为np.nan, scale也已处理过"

    # 最后记得关闭文件
    new_NC.close()
# %% 原数据去趋势 Z-score
# gpp = exact_data1()
# gpp_de = detrend(gpp)
# gpp_z = Zscore(gpp_de)
# CreatNC(gpp_de, gpp_z)


# %% 原数据Z-Score
# gpp = exact_data1()
# gpp_z = Zscore(gpp)
# CreatNC2(gpp_z)

# %% 生成
gpp = exact_data1()
CreatNC3(gpp)
# gpp_gsl = np.nanmean(gpp[:, 3:10, :, :], axis=1)
