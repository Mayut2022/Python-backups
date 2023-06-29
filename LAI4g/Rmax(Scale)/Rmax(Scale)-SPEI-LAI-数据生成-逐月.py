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
    inpath = (r"E:/LAI4g/data_MG/LAI_Zscore_82_20_MG_SPEI0.5x0.5.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        # print(f.variables['lai'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        lai = (f.variables['lai'][:, 4:9, :, :])
        lai_gsl = lai.reshape(39*5, 30, 50)

    return lai_gsl


# %%


def read_nc2(scale):
    inpath = (rf"E:/SPEI_base/data/spei{scale}.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        spei = (f.variables['spei'][972:])
        time = (f.variables['time'][960:])
        t2 = nc.num2date(time, 'days since 1900-01-01 00:00:0.0').data
        lat_g = (f.variables['lat'][:])
        lon_g = (f.variables['lon'][:])
    ################

    def mn_yr(data):
        tmp_mn = []
        for mn in range(12):
            tmp_ = []
            for yr in range(39):
                tmp_.append(data[mn])
                mn += 12
            tmp_mn.append(tmp_)

        tmp_mn = np.array(tmp_mn)

        return tmp_mn
    #################

    def sif_xarray(band1):
        mn = np.arange(12)
        yr = np.arange(39)
        sif = xr.DataArray(band1, dims=['mn', 'yr', 'y', 'x'], coords=[
                           mn, yr, lat_g, lon_g])

        sif_MG = sif.loc[:, :, 40:55, 100:125]
        return np.array(sif_MG)

    spei_mn = mn_yr(spei)
    spei_mn_MG = sif_xarray(spei_mn)
    spei_gsl = spei_mn_MG[4:9, :].reshape((39*5, 30, 50), order="F")
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


# %% Rmax\Rmax_scale 筛选最大绝对值


def Rmax_RmaxScale(corr):
    Rmax = np.zeros((195, 30, 50))
    Rmax_scale = np.zeros((195, 30, 50))
    for k in range(195):
        for i in range(30):
            for j in range(50):
                if np.isnan(corr[:, k, i, j]).any() or np.isnan(corr[:, k, i, j]).any():
                    Rmax[k, i, j], Rmax_scale[k, i, j] = np.nan, np.nan
                else:
                    r = corr[:, k, i, j]
                    if r.mean() >= 0:
                        Rmax[k, i, j] = r.max()
                        Rmax_scale[k, i, j] = r.argmax()+1
                    else:
                        Rmax[k, i, j] = r.min()
                        Rmax_scale[k, i, j] = r.argmin()+1

    return Rmax, Rmax_scale


# %%生成新的nc文件
def CreatNC(windows, data1, data2):
    new_NC = nc.Dataset(
        rf"E:/LAI4g/data_MG/Rmax(Scale)/LAI_SPEI_MG_MonthWindows{windows}.nc", 'w', format='NETCDF4')

    time = np.arange(195)

    new_NC.createDimension('time', 195)
    new_NC.createDimension('lat', 30)
    new_NC.createDimension('lon', 50)

    var = new_NC.createVariable('Rmax', 'f', ("time", "lat", "lon"))
    new_NC.createVariable('Rmax_scale', 'f', ("time", "lat", "lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))

    new_NC.variables['Rmax'][:] = data1
    new_NC.variables['Rmax_scale'][:] = data2
    new_NC.variables['lat'][:] = lat
    new_NC.variables['lon'][:] = lon

    # 最后记得关闭文件
    new_NC.close()


# %% 数据读取
lcc = read_lcc()
lai = read_nc()

scale = [str(x).zfill(2) for x in range(1, 13)]

for win in [10, 20, 30, 40, 50][3:]:
    corr = []
    for s in scale[:]:
        spei = read_nc2(s)
        r = roll_corr(lai, spei, win)
        corr.append(r)
        print(f"scale spei{s} is done!")
    corr = np.array(corr)
    Rmax, Rmax_scale = Rmax_RmaxScale(corr)
    CreatNC(win, Rmax, Rmax_scale)




# %% 格点测试
# spei01 = read_nc2(scale[0])
# a = pd.Series(lai[:, 10, 10], name="LAI")
# b = pd.Series(spei01[:, 10, 10], name="SPEI")
# df = pd.concat([a, b], axis=1)
# r = df.rolling(10).corr().iloc[1::2, 0]
