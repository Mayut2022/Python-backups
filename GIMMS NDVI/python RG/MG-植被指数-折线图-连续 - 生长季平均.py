# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 10:24:03 2022

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

from scipy.stats import linregress
from sklearn import preprocessing
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
    global spei, t
    inpath = (rf"E:/SPEI_base/data/spei03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())

        spei = (f.variables['spei'][960:])


# %%
def read_nc3():
    global ndvi, lat2, lon2
    inpath = r"E:/GIMMS_NDVI/data_RG/NDVI_82_15_RG_SPEI0.5x0.5.nc"
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())

        ndvi = (f.variables['ndvi'][:, 4:9, :])
        lat2 = (f.variables['lat'][:])
        lon2 = (f.variables['lon'][:])
        ndvi = np.nanmean(ndvi, axis=1)

# %%


def read_nc4():
    global sif
    inpath = r"E:/Gosif_Monthly/data_RG/GOSIF_01_20_RG_SPEI0.5X0.5.nc"
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())

        sif = (f.variables['sif'][:, 4:9, :])
        lat2 = (f.variables['lat'][:])
        lon2 = (f.variables['lon'][:])
        sif = np.nanmean(sif, axis=1)

# %%


def read_nc5():
    global gpp
    inpath = rf"E:/GLASS-GPP/MG detrend Zscore/MG_GPP_Zscore_SPEI_0.5X0.5.nc"
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())

        gpp = (f.variables['GPP_Z'][:, 4:9, :])
        gpp = np.nanmean(gpp, axis=1)

# %%


def ndvi_xarray(band1):
    t = np.arange(34)
    ndvi = xr.DataArray(band1, dims=['t', 'y', 'x'], coords=[
                        t, lat2, lon2])
    ndvi_IM = ndvi.loc[:, 40:55, 100:125]
    lat_IM = ndvi_IM.y
    lon_IM = ndvi_IM.x
    ndvi_IM = np.array(ndvi_IM)
    return ndvi_IM, lat_IM, lon_IM

# %%


def sif_xarray(band1):
    yr = np.arange(20)
    sif = xr.DataArray(band1, dims=['yr', 'y', 'x'], coords=[
                       yr, lat2, lon2])

    sif_MG = sif.loc[:, 40:55, 100:125]
    return np.array(sif_MG)

# %% mask数组


def mask(x, data):
    lcc2 = np.array(lcc)

    np.place(lcc2, lcc2 != x, 1)
    np.place(lcc2, lcc2 == x, 0)

    spei_ma = np.empty((37, 30, 50))
    spei_ma = np.ma.array(spei_ma)

    for i in range(12):
        a = data[i, :, :]
        a = ma.masked_array(a, mask=lcc2)
        spei_ma[i, :, :] = a

    spei_ma_ave = np.nanmean(spei_ma, axis=(1, 2))

    return spei_ma_ave

# %% mask数组


def mask2(x, data):
    lcc2 = np.array(lcc)

    np.place(lcc2, lcc2 != x, 1)
    np.place(lcc2, lcc2 == x, 0)

    spei_ma = np.empty((34, 30, 50))
    spei_ma = np.ma.array(spei_ma)

    for i in range(34):
        a = data[i, :, :]
        a = ma.masked_array(a, mask=lcc2)
        spei_ma[i, :, :] = a

    spei_ma_ave = np.nanmean(spei_ma, axis=(1, 2))

    return spei_ma_ave

# %% mask数组


def mask3(x, data):
    lcc2 = np.array(lcc)

    np.place(lcc2, lcc2 != x, 1)
    np.place(lcc2, lcc2 == x, 0)

    spei_ma = np.empty((20, 30, 50))
    spei_ma = np.ma.array(spei_ma)

    for i in range(20):
        a = data[i, :, :]
        a = ma.masked_array(a, mask=lcc2)
        spei_ma[i, :, :] = a

    spei_ma_ave = np.nanmean(spei_ma, axis=(1, 2))

    return spei_ma_ave


# %% mask数组


def mask4(x, data):
    lcc2 = np.array(lcc)

    np.place(lcc2, lcc2 != x, 1)
    np.place(lcc2, lcc2 == x, 0)

    spei_ma = np.empty((37, 30, 50))
    spei_ma = np.ma.array(spei_ma)

    for i in range(37):
        a = data[i, :, :]
        a = ma.masked_array(a, mask=lcc2)
        spei_ma[i, :, :] = a

    spei_ma_ave = np.nanmean(spei_ma, axis=(1, 2))

    return spei_ma_ave


# %% NDVI 标准化
def Zscore(data):
    data_z = np.zeros((34, 30, 50))
    for r in range(30):
        for c in range(50):
            data_z[:, r, c] = preprocessing.scale(data[:, r, c])

    return data_z

# %% SIF 标准化


def Zscore2(data):
    data_z = np.zeros((20, 30, 50))
    for r in range(30):
        for c in range(50):
            data_z[:, r, c] = preprocessing.scale(data[:, r, c])

    return data_z

# %%


def subplot(data1, data2, data3, ax, sy, df):
    t = pd.date_range(f'{sy}', periods=240, freq="MS")
    # ax.axhline(y=std, c="orange", linestyle="--")
    # ax.axhline(y=-std, c="orange", linestyle="--")
    ax.axhline(y=0, c="k", linestyle="--")

    for y1, m1, y2, m2 in zip(df["SY"], df["SM"], df["EY"], df["EM"]):
        d1 = dt.datetime(y1, m1, 1, 0, 0, 0)
        d2 = dt.datetime(y2, m2, 1, 0, 0, 0)+dt.timedelta(days=35)
        ax.fill_between([d1, d2], 2.8, -2.8,
                        facecolor='dodgerblue', alpha=0.4)  # 貌似只要是时间格式的x都行

    ax.scatter(t, data1, c='orange', s=20)
    ax.plot(t, data1, c='orange', label="SIF Zscore", linewidth=2)
    ax.scatter(t, data2, c='Green', s=20)
    ax.plot(t, data2, c='Green', label="NDVI Zscore", linewidth=2)
    ax.scatter(t, data3, color='blue', s=20)
    ax.plot(t, data3, color='blue', label="GPP Zscore", linewidth=2)

    ax.set_ylim(-2.8, 2.8)
    ax.set_yticks(np.arange(-2, 2.1, 1))

    mn = pd.to_timedelta("100 days")
    ax.set_xlim(t[0]-mn, t[-1]+mn)

    tt = pd.date_range(f'{sy}', periods=10, freq='2AS-JAN')
    ax.set_xticks(tt)
    ax.xaxis.set_major_formatter(mdate.DateFormatter('%y-%m'))
    ax.tick_params(labelsize=20)
    ax.tick_params(which="major", bottom=1, left=1, length=8)

    return ax


def plot(data1, data2, data3, title):
    fig = plt.figure(1, figsize=(10, 4), dpi=500)

    fig.subplots_adjust(left=0.05, bottom=0.1, right=0.95,
                        top=0.9, wspace=None, hspace=0.12)

    t = pd.date_range(f'1981', periods=40, freq="YS")
    ax = fig.subplots(1)
    ax.axhline(y=0, c="k", linestyle="--")

    ax.scatter(t[20:], data1, c='orange', s=20)
    ax.plot(t[20:], data1, c='orange', label="SIF Zscore", linewidth=2)
    ax.scatter(t[1:35], data2, c='Green', s=20)
    ax.plot(t[1:35], data2, c='Green', label="NDVI Zscore", linewidth=2)
    ax.scatter(t[1:38], data3, color='blue', s=20)
    ax.plot(t[1:38], data3, color='blue', label="GPP Zscore", linewidth=2)

    plt.legend(loc='lower right', fontsize=15)
    plt.xlabel("years", fontsize=15)
    plt.suptitle(f'{title}', fontsize=20)
    # plt.savefig(rf'E:/GIMMS_NDVI/JPG_RG/{title}.jpg',
    #           bbox_inches='tight')
    plt.show()


# %%
read_nc()  # lcc

# %% NDVI
read_nc3()  # ndvi
ndvi_IM, _, _ = ndvi_xarray(ndvi)

# NDVI标准化
ndvi_IM_z = Zscore(ndvi_IM)
ndvi_IM_z_ave = mask2(130, ndvi_IM_z)
print(np.nanmax(ndvi_IM_z_ave), np.nanmin(ndvi_IM_z_ave))


# %% SIF 数据读取
read_nc4()  # sif
sif_MG = sif_xarray(sif)
sif_MG_z = Zscore2(sif_MG)
sif_MG_z_ave = mask3(130, sif_MG_z)
print(np.nanmax(sif_MG_z_ave), np.nanmin(sif_MG_z_ave))


# %% GPP数据读取
read_nc5()  # gpp
gpp_ave = mask4(130, gpp)
print(np.nanmax(gpp_ave), np.nanmin(gpp_ave))


# %%

# df = pd.read_excel(
#     "E:/SPEI_base/python MG/MG Grassland干旱统计.xlsx", sheet_name='Grassland')

plot(sif_MG_z_ave, ndvi_IM_z_ave, gpp_ave, f"MG Grassland Line GSL")

# %%


def trend(data, var):
    t = np.arange(len(data))
    s, _, _, p, _ = linregress(t, data)

    print(var)
    print("slope:", s, "p-value:", p, "\n")


trend(sif_MG_z_ave, "SIF 01-20")
trend(ndvi_IM_z_ave, "NDVI 82-15")
trend(gpp_ave, "GPP 82-18")

# %% 82-00
trend(ndvi_IM_z_ave[:19], "NDVI 82-00")
trend(gpp_ave[:19], "GPP 82-00")

# %% 00-18
trend(ndvi_IM_z_ave[19:], "NDVI 01-15")
trend(gpp_ave[19:], "GPP 01-18")