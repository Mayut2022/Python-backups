# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 12:53:27 2023

@author: MaYutong
"""
import netCDF4 as nc
import numpy as np
import pandas as pd
import xarray as xr


# %%
def read_nc():
    global lcc, lat, lon
    # global a1, a2, o1, o2
    inpath = (r"E:\ESA\ESA_LUC_2000_SPEI03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        # print(f.variables['lccs_class'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        lcc = (f.variables['lcc'][:])


# %%
def read_nc2():
    global pre_anom, lat2, lon2
    inpath = r"E:/CRU/Q_DATA_CRU-GLEAM/PRE_global_MONTH_ANOM_81_20.nc"
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())

        pre_anom = (f.variables['pre'][:])
        lat2 = (f.variables['lat'][:])
        lon2 = (f.variables['lon'][:])


# %% 480 -> 月 年
def mn_yr(data):
    q_mn = []
    for mn in range(12):
        q_ = []
        for yr in range(40):
            q_.append(data[mn])
            mn += 12
        q_mn.append(q_)

    q_mn = np.array(q_mn)
    return q_mn

# %%


def pre_xarray(band1):
    t = np.arange(40)
    mn = np.arange(1, 13, 1)
    pre = xr.DataArray(band1, dims=['mn', 't', 'y', 'x'], coords=[
                       mn, t, lat2, lon2])
    pre_MG = pre.loc[:, :, 40:55, 100:125]
    lat_MG = pre_MG.y
    lon_MG = pre_MG.x
    pre_MG = np.array(pre_MG)
    return pre_MG, lat_MG, lon_MG

# %% mask数组


def mask(x, data):
    lcc2 = np.array(lcc)

    np.place(lcc2, lcc2 != x, 1)
    np.place(lcc2, lcc2 == x, 0)

    spei_ma = np.empty((12, 40, 30, 50))
    spei_ma = np.ma.array(spei_ma)

    for i in range(12):
        for l in range(40):
            a = data[i, l, :, :]
            a = np.ma.masked_array(a, mask=lcc2)
            spei_ma[i, l, :, :] = a

    spei_ma_ave = np.nanmean(spei_ma, axis=(2, 3))

    return spei_ma_ave


# %%
read_nc()
read_nc2()
preanom = mn_yr(pre_anom)
preanom_MG, lat_MG, lon_MG = pre_xarray(preanom)
preanom_ave = mask(130, preanom_MG)
