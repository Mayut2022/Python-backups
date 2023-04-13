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

        ndvi = (f.variables['ndvi'][:])
        lat2 = (f.variables['lat'][:])
        lon2 = (f.variables['lon'][:])

# %%


def read_nc4():
    global sif
    inpath = r"E:/Gosif_Monthly/data_RG/GOSIF_01_20_RG_SPEI0.5X0.5.nc"
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())

        sif = (f.variables['sif'][:])
        lat2 = (f.variables['lat'][:])
        lon2 = (f.variables['lon'][:])

# %%


def read_nc5():
    global gpp
    inpath = rf"E:/GLASS-GPP/MG detrend Zscore/MG_GPP_Zscore_SPEI_0.5X0.5.nc"
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())

        gpp = (f.variables['GPP_Z'][:])

# %%


def ndvi_xarray(band1):
    t = np.arange(34)
    mn = np.arange(1, 13, 1)
    ndvi = xr.DataArray(band1, dims=['t', 'mn', 'y', 'x'], coords=[
                        t, mn, lat2, lon2])
    ndvi_IM = ndvi.loc[:, :, 40:55, 100:125]
    lat_IM = ndvi_IM.y
    lon_IM = ndvi_IM.x
    ndvi_IM = np.array(ndvi_IM)
    return ndvi_IM, lat_IM, lon_IM

# %%


def sif_xarray(band1):
    yr = np.arange(20)
    mn = np.arange(12)
    sif = xr.DataArray(band1, dims=['yr', 'mn', 'y', 'x'], coords=[
                       yr, mn, lat2, lon2])

    sif_MG = sif.loc[:, :, 40:55, 100:125]
    return np.array(sif_MG)

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
            a = ma.masked_array(a, mask=lcc2)
            spei_ma[i, l, :, :] = a

    spei_ma_ave = np.nanmean(spei_ma, axis=(2, 3))

    return spei_ma_ave

# %% mask数组


def mask2(x, data):
    lcc2 = np.array(lcc)

    np.place(lcc2, lcc2 != x, 1)
    np.place(lcc2, lcc2 == x, 0)

    spei_ma = np.empty((34, 12, 30, 50))
    spei_ma = np.ma.array(spei_ma)

    for i in range(34):
        for l in range(12):
            a = data[i, l, :, :]
            a = ma.masked_array(a, mask=lcc2)
            spei_ma[i, l, :, :] = a

    spei_ma_ave = np.nanmean(spei_ma, axis=(2, 3))

    return spei_ma_ave

# %% mask数组


def mask3(x, data):
    lcc2 = np.array(lcc)

    np.place(lcc2, lcc2 != x, 1)
    np.place(lcc2, lcc2 == x, 0)

    spei_ma = np.empty((20, 12, 30, 50))
    spei_ma = np.ma.array(spei_ma)

    for i in range(20):
        for l in range(12):
            a = data[i, l, :, :]
            a = ma.masked_array(a, mask=lcc2)
            spei_ma[i, l, :, :] = a

    spei_ma_ave = np.nanmean(spei_ma, axis=(2, 3))

    return spei_ma_ave


# %% mask数组


def mask4(x, data):
    lcc2 = np.array(lcc)

    np.place(lcc2, lcc2 != x, 1)
    np.place(lcc2, lcc2 == x, 0)

    spei_ma = np.empty((37, 12, 30, 50))
    spei_ma = np.ma.array(spei_ma)

    for i in range(37):
        for l in range(12):
            a = data[i, l, :, :]
            a = ma.masked_array(a, mask=lcc2)
            spei_ma[i, l, :, :] = a

    spei_ma_ave = np.nanmean(spei_ma, axis=(2, 3))

    return spei_ma_ave

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


# %% NDVI 标准化
def Zscore(data):
    data_z = np.zeros((34, 12, 30, 50))
    for mn in range(12):
        for r in range(30):
            for c in range(50):
                data_z[:, mn, r, c] = preprocessing.scale(data[:, mn, r, c])

    return data_z

# %% SIF 标准化


def Zscore2(data):
    data_z = np.zeros((20, 12, 30, 50))
    for mn in range(12):
        for r in range(30):
            for c in range(50):
                data_z[:, mn, r, c] = preprocessing.scale(data[:, mn, r, c])

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


def plot(data1, data2, data3, df, title):
    fig = plt.figure(1, figsize=(22, 12), dpi=500)

    fig.subplots_adjust(left=0.05, bottom=0.1, right=0.95,
                        top=0.93, wspace=None, hspace=0.12)
    axs = fig.subplots(2, 1, sharey=True)

    a, b, c = np.zeros(480), np.zeros(480), np.zeros(480)
    a[0:240], a[240:480] = np.nan, data1
    b[0:12], b[12:420], b[420:] = np.nan, data2, np.nan
    c[0:12], c[12:456], c[456:] = np.nan, data3, np.nan

    data1 = a
    data2 = b
    data3 = c

    axs[0] = subplot(data1[:240], data2[:240], data3[:240], axs[0], 1981, df)
    axs[1] = subplot(data1[240:], data2[240:], data3[240:], axs[1], 2001, df)

    plt.legend(loc='lower right', fontsize=20)
    plt.xlabel("years", fontsize=25)
    plt.suptitle(f'{title}', fontsize=30)
    plt.savefig(rf'E:/GIMMS_NDVI/JPG_RG/{title}.jpg',
              bbox_inches='tight')
    plt.show()


# %%
read_nc()

read_nc3()
ndvi_IM, _, _ = ndvi_xarray(ndvi)

ndvi_IM_ave = mask2(130, ndvi_IM)


print(np.nanmax(ndvi_IM_ave), np.nanmin(ndvi_IM_ave))

# NDVI标准化
ndvi_IM_z = Zscore(ndvi_IM)
ndvi_IM_z_ave = mask2(130, ndvi_IM_z)
print(np.nanmax(ndvi_IM_z_ave), np.nanmin(ndvi_IM_z_ave))

# %% reshape (34,12) -> (408)
ndvi_re = ndvi_IM_ave.reshape(408)
ndvi_z_re = ndvi_IM_z_ave.reshape(408)
print(np.nanmax(ndvi_z_re), np.nanmin(ndvi_z_re))

# %% SIF 数据读取
read_nc4()
sif_MG = sif_xarray(sif)
sif_MG_z = Zscore2(sif_MG)
sif_MG_z_ave = mask3(130, sif_MG_z)
sif_re = sif_MG_z_ave.reshape(240)
print(np.nanmax(sif_re), np.nanmin(sif_re))


# %% GPP数据读取
read_nc5()
gpp_ave = mask4(130, gpp)
gpp_re = gpp_ave.reshape(444)
print(np.nanmax(gpp_re), np.nanmin(gpp_re))
# %%

df = pd.read_excel(
    "E:/SPEI_base/python MG/MG Grassland干旱统计.xlsx", sheet_name='Grassland')

plot(sif_re, ndvi_z_re, gpp_re, df, f"MG Grassland Line")
