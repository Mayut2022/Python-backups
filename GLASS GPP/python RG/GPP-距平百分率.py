# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 21:42:33 2023

@author: MaYutong
"""

# %%
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
# from sklearn import preprocessing
import xarray as xr

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['times new roman']  # 指定默认字体
# %%


def read_nc():
    global lcc, lat, lon
    # global a1, a2, o1, o2
    inpath = (r"E:/ESA/ESA_LUC_2000_SPEI03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        # print(f.variables['lccs_class'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        lcc = (f.variables['lcc'][:])

# %%


def sif_xarray(band1):
    mn = np.arange(12)
    sif = xr.DataArray(band1, dims=['mn', 'y', 'x'], coords=[mn, lat2, lon2])

    sif_MG = sif.loc[:, 40:55, 100:125]
    # sif_MG_gsl = np.nanmean(sif_MG[3:10, :, :], axis=0)
    return np.array(sif_MG)


def sif_xarray_gsl(band1):
    mn = np.arange(12)
    sif = xr.DataArray(band1, dims=['mn', 'y', 'x'], coords=[mn, lat2, lon2])

    sif_MG = sif.loc[:, 40:55, 100:125]
    sif_MG_gsl = np.nanmean(sif_MG[3:10, :, :], axis=0)
    return np.array(sif_MG_gsl)


# %% mask数组

def mask(x, data):
    lcc2 = np.array(lcc)

    np.place(lcc2, lcc2 != x, 1)
    np.place(lcc2, lcc2 == x, 0)

    spei_ma = np.empty((12, 30, 50))
    spei_ma = np.ma.array(spei_ma)

    for l in range(12):
        a = data[l, :, :]
        a = ma.masked_array(a, mask=lcc2)
        spei_ma[l, :, :] = a

    spei_ma_ave = np.nanmean(spei_ma, axis=(1, 2))

    return spei_ma_ave


def mask_gsl(x, data):
    lcc2 = np.array(lcc)

    np.place(lcc2, lcc2 != x, 1)
    np.place(lcc2, lcc2 == x, 0)

    spei_ma = ma.masked_array(data, mask=lcc2)
    spei_ma_ave = np.nanmean(spei_ma)

    return spei_ma_ave

# %%


def read_nc2(inpath):
    global lat2, lon2
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())

        gpp = (f.variables['GPP'][:])
        lat2 = (f.variables['lat'][:])
        lon2 = (f.variables['lon'][:])

    return gpp


def read_nc3(inpath):
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        gpp_a = (f.variables['GPP'][:])
        lat2 = (f.variables['lat'][:])
        lon2 = (f.variables['lon'][:])

    return gpp_a


def exact_data1():
    for yr in range(1982, 2019):
        inpath = rf"E:/GLASS-GPP/Month RG/GLASS_GPP_RG_SPEI_0.5X0.5_{yr}.nc"
        data = read_nc2(inpath)

        data_MG = sif_xarray(data)
        data_ave = mask(130, data_MG)
        data_MG_gsl = sif_xarray_gsl(data)
        data_ave_gsl = mask_gsl(130, data_MG_gsl)
        if yr == 1982:
            data_all = data_ave
            data_all_gsl = data_ave_gsl
        else:
            data_all = np.hstack((data_all, data_ave))
            data_all_gsl = np.hstack((data_all_gsl, data_ave_gsl))
    data_all = data_all.reshape(37, 12)

    return data_all, data_all_gsl


def exact_data2():
    for yr in range(1982, 2019):
        inpath2 = rf"E:/GLASS-GPP/Month RG/GLASS_GPP_RG_Anomaly_SPEI_0.5X0.5_{yr}.nc"
        data = read_nc3(inpath2)

        data_MG = sif_xarray(data)
        data_ave = mask(130, data_MG)
        data_MG_gsl = sif_xarray_gsl(data)
        data_ave_gsl = mask_gsl(130, data_MG_gsl)
        if yr == 1982:
            data_all = data_ave
            data_all_gsl = data_ave_gsl
        else:
            data_all = np.hstack((data_all, data_ave))
            data_all_gsl = np.hstack((data_all_gsl, data_ave_gsl))
    data_all = data_all.reshape(37, 12)

    return data_all, data_all_gsl

# %%


def plot(data1, title):
    print(np.nanmax(data1), np.nanmin(data1))

    fig, axs = plt.subplots(1, 3, figsize=(
        16, 4), dpi=500, sharey=True, sharex=True)
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9,
                        top=0.85, wspace=0.08, hspace=0.15)

    t = np.arange(1982, 2019)
    t1 = np.arange(1982, 2000)
    t2 = np.arange(2000, 2019)
    mn = ["June", "July", "Augest"]

    for i in range(3):
        axs[i].fill_between([1999, 2019], -1, 1, facecolor='dodgerblue', alpha=0.2)
        axs[i].bar(t, data1[:, i], color='orange', width=0.8, alpha=0.5)

        axs[i].set_xlim(1980, 2019)
        axs[i].set_xticks(np.arange(1981, 2019, 5))
        axs[i].set_ylim(-0.25, 0.25)
        yticks = np.arange(-0.2, 0.21, 0.1)
        axs[i].set_yticks(yticks)
        ylabels = [f"{100*x:.0f}%" for x in yticks]
        axs[i].set_yticklabels(ylabels)
        axs[i].set_title(f"{mn[i]}", fontsize=15, loc="left")
        axs[i].tick_params(labelsize=15)

    axs[0].set_ylabel("Percentage of anomaly", fontsize=15)

    # plt.suptitle(f"{title}", fontsize=25)
    plt.savefig(rf'E:/GLASS-GPP/JPG MG/{title}.jpg',
                bbox_inches='tight')
    plt.show()


# %%
def plot2(data1, title):
    print(np.nanmax(data1), np.nanmin(data1))

    fig = plt.figure(1, dpi=500, figsize=(7, 5))
    axs = fig.add_subplot(111)
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9,
                        top=0.85, wspace=0.08, hspace=0.15)

    t = np.arange(1982, 2019)
    t1 = np.arange(1981, 2000)
    t2 = np.arange(2000, 2021)
    mn = ["July", "Augest"]

    axs.bar(t, data1, color='orange', width=0.8, alpha=0.5)

    axs.set_xlim(1980, 2019)
    axs.set_xticks(np.arange(1981, 2019, 5))
    axs.set_ylim(-0.25, 0.25)
    yticks = np.arange(-0.2, 0.21, 0.1)
    axs.set_yticks(yticks)
    ylabels = [f"{100*x:.0f}%" for x in yticks]
    axs.set_yticklabels(ylabels)
    axs.tick_params(labelsize=15)

    axs.set_ylabel("Percentage of anomaly", fontsize=15)

    # plt.suptitle(f"{title}", fontsize=25)
    plt.savefig(rf'E:/GLASS-GPP/JPG MG/{title}.jpg',
                bbox_inches='tight')
    plt.show()

# %% 82-99/00-18


def segment_ave(data):
    d1, d2 = data[:18, :].mean(axis=0), data[18:, :].mean(axis=0)
    d_gsl1, d_gsl2 = data[:18, 3:10].mean(), data[18:, 3:10].mean()
    return d1, d2, d_gsl1, d_gsl2


# %%
read_nc()
gpp, gpp_gsl = exact_data1()
gpp_a, gpp_a_gsl = exact_data2()
# gpp_pa, gpp_pa_gsl = gpp_a/gpp, gpp_a_gsl/gpp_gsl

gpp1, gpp2, gpp_gsl1, gpp_gsl2 = segment_ave(gpp)
# %%
gpp_pa = np.zeros((37, 12))  # 距平百分率
gpp_yr = np.nanmean(gpp, axis=0)  # q_ave的逐月多年平均
for yr in range(37):
    gpp_pa[yr, :] = gpp_a[yr, :]/gpp_yr[:]

plot(gpp_pa[:, 5:8], title=r"MG Grassland GPP Percentage of anomaly Summer")

gpp_pa_gsl = gpp_a_gsl/gpp_gsl
# plot2(gpp_pa_gsl, title=r"MG Grassland GPP Percentage of anomaly GSL")

