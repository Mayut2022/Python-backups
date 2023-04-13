# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 11:13:17 2022

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
import xarray as xr

plt.rcParams['font.sans-serif'] = ['times new roman']  # 指定默认字体

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
    global q
    inpath = r"E:/CRU/Q_DATA_CRU-GLEAM/Q_global_81_20.nc"
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())

        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        q = (f.variables['Q'][:])


# %%
def read_nc3():
    global q_anom, lat2, lon2
    inpath = r"E:/CRU/Q_DATA_CRU-GLEAM/Q_global_MONTH_ANOM_81_20.nc"
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())

        q_anom = (f.variables['Q'][:])
        lat2 = (f.variables['lat'][:])
        lon2 = (f.variables['lon'][:])

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


# %%
def plot(data1, title):
    print(np.nanmax(data1), np.nanmin(data1))

    fig, axs = plt.subplots(1, 2, figsize=(
        14, 5), dpi=500, sharey=True, sharex=True)
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9,
                        top=0.85, wspace=0.08, hspace=0.15)

    t = np.arange(1981, 2021)
    t1 = np.arange(1981, 2000)
    t2 = np.arange(2000, 2021)
    mn = ["July", "Augest"]

    for i in range(2):
        axs[i].bar(t, data1[i, :], color='Green', width=0.8, alpha=0.5)

        axs[i].set_xlim(1980, 2021)
        axs[i].set_xticks(np.arange(1980, 2021, 5))
        axs[i].set_ylim(-0.32, 0.32)
        yticks = np.arange(-0.3, 0.31, 0.1)

        axs[i].set_yticks(yticks)
        ylabels = [f"{100*x:.0f}%" for x in yticks]
        axs[i].set_yticklabels(ylabels)
        axs[i].set_title(f"{mn[i]}", fontsize=15, loc="left")
        axs[i].tick_params(labelsize=15)

    axs[0].set_ylabel("Percentage of anomaly", fontsize=15)

    plt.suptitle(f"{title}", fontsize=25)
    plt.savefig(rf'E:/CRU/JPG_MG/{title}.jpg',
                bbox_inches='tight')
    plt.show()


# %%
def plot2(data1, title):
    print(np.nanmax(data1), np.nanmin(data1))

    fig = plt.figure(1, dpi=500, figsize=(7, 5))
    axs = fig.add_subplot(111)
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9,
                        top=0.85, wspace=0.08, hspace=0.15)

    t = np.arange(1981, 2021)
    t1 = np.arange(1981, 2000)
    t2 = np.arange(2000, 2021)
    mn = ["July", "Augest"]

    axs.bar(t, data1, color='Green', width=0.8, alpha=0.5)

    axs.set_xlim(1980, 2021)
    axs.set_xticks(np.arange(1980, 2021, 5))
    axs.set_ylim(-1.36, 1.36)
    yticks = np.arange(-1.2, 1.21, 0.3)
    axs.set_yticks(yticks)
    ylabels = [f"{100*x:.0f}%" for x in yticks]
    axs.set_yticklabels(ylabels)
    axs.tick_params(labelsize=15)

    axs.set_ylabel("Percentage of anomaly", fontsize=15)

    plt.suptitle(f"{title}", fontsize=25)
    plt.savefig(rf'E:/CRU/JPG_MG/{title}.jpg',
                bbox_inches='tight')
    plt.show()

# %% 81-99/00-20


def segment_ave(data):
    d1, d2 = data[:, :19].mean(axis=1), data[:, 19:].mean(axis=1)
    d_gsl1, d_gsl2 = data[3:10, :19].mean(), data[3:10, 19:].mean()
    return d1, d2, d_gsl1, d_gsl2


# %%
read_nc()
read_nc2()
read_nc3()

q_mn = mn_yr(q)
q_MG, _, _ = pre_xarray(q_mn)
q_ave = mask(130, q_MG)


qanom = mn_yr(q_anom)
qanom_MG, _, _ = pre_xarray(qanom)
qanom_ave = mask(130, qanom_MG)

# %%
q_pa = np.zeros((12, 40))  # 距平百分率
q_a = np.nanmean(q_ave, axis=1)  # q_ave的逐月多年平均
for yr in range(40):
    q_pa[:, yr] = qanom_ave[:, yr]/q_a[:]

q_pa_gsl = np.nanmean(q_pa[3:10, :], axis=0)
q1, q2, q_gsl1, q_gsl2 = segment_ave(q_ave)
# plot(q_pa[6:8, :], title=r"MG Grassland Water Yield Percentage of anomaly")
# plot2(q_pa_gsl, title=r"MG Grassland Water Yield Percentage of anomaly GSL")
