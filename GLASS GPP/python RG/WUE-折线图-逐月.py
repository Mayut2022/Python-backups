# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 10:35:02 2022

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

# %%


def read_nc():
    global lcc, lat, lon
    # global a1, a2, o1, o2
    inpath = (r"/mnt/e/ESA/ESA_LUC_2000_SPEI03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        # print(f.variables['lccs_class'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        lcc = (f.variables['lcc'][:])

# %%


def read_nc2(inpath):
    global gpp, lat2, lon2
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())

        gpp = (f.variables['GPP'][:])
        lat2 = (f.variables['lat'][:])
        lon2 = (f.variables['lon'][:])

    return gpp

# %%


def read_nc3(inpath):
    global gpp_a
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        print(f.variables['GPP'])
        gpp_a = (f.variables['GPP'][:])
        lat2 = (f.variables['lat'][:])
        lon2 = (f.variables['lon'][:])

    return gpp_a

# %%


def read_nc4(inpath):
    global wue
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())

        wue = (f.variables['WUE'][:])
        lat2 = (f.variables['lat'][:])
        lon2 = (f.variables['lon'][:])

    return wue

# %%


def read_nc5(inpath):
    global wue_a
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())

        wue_a = (f.variables['WUE'][:])
        lat2 = (f.variables['lat'][:])
        lon2 = (f.variables['lon'][:])

    return wue_a

# %%


def sif_xarray(band1):
    mn = np.arange(12)
    sif = xr.DataArray(band1, dims=['mn', 'y', 'x'], coords=[mn, lat2, lon2])

    sif_MG = sif.loc[:, 40:55, 100:125]
    return np.array(sif_MG)

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


# %%
read_nc()


def exact_data1():
    for yr in range(1982, 2019):
        inpath = rf"/mnt/e/GLASS-GPP/Month RG/GLASS_GPP_RG_SPEI_0.5X0.5_{yr}.nc"
        data = read_nc2(inpath)

        data_MG = sif_xarray(data)
        data_ave = mask(130, data_MG)
        if yr == 1982:
            data_all = data_ave
        else:
            data_all = np.hstack((data_all, data_ave))
    data_all = data_all.reshape(37, 12)
    return data_all


def exact_data2():
    for yr in range(1982, 2019):
        inpath2 = rf"/mnt/e/GLASS-GPP/Month RG/GLASS_GPP_RG_Anomaly_SPEI_0.5X0.5_{yr}.nc"
        data = read_nc3(inpath2)

        data_MG = sif_xarray(data)
        data_ave = mask(130, data_MG)
        if yr == 1982:
            data_all = data_ave
        else:
            data_all = np.hstack((data_all, data_ave))
    data_all = data_all.reshape(37, 12)
    return data_all


def exact_data3():
    for yr in range(1982, 2019):
        inpath3 = rf"/mnt/e/GLASS-GPP/Month RG WUE/GLASS_WUE_RG_SPEI_0.5X0.5_{yr}.nc"
        data = read_nc4(inpath3)

        data_MG = sif_xarray(data)
        data_ave = mask(130, data_MG)
        if yr == 1982:
            data_all = data_ave
        else:
            data_all = np.hstack((data_all, data_ave))
    data_all = data_all.reshape(37, 12)
    return data_all


def exact_data4():
    for yr in range(1982, 2019):
        inpath4 = rf"/mnt/e/GLASS-GPP/Month RG WUE/GLASS_WUE_RG_Anomaly_SPEI_0.5X0.5_{yr}.nc"
        data = read_nc5(inpath4)

        data_MG = sif_xarray(data)
        data_ave = mask(130, data_MG)
        if yr == 1982:
            data_all = data_ave
        else:
            data_all = np.hstack((data_all, data_ave))
    data_all = data_all.reshape(37, 12)
    return data_all


# %% GPP
gpp_MG = exact_data1()
gpp_a_MG = exact_data2()
wue_MG = exact_data3()
wue_a_MG = exact_data4()


# %%
def plot(data1, data2, title):
    print(np.nanmax(data1), np.nanmin(data1))
    print(np.nanmax(data2), np.nanmin(data2))

    fig, axs = plt.subplots(4, 3, figsize=(20, 12), dpi=150, sharey=True)
    fig.subplots_adjust(left=0.05, bottom=0.1, right=0.95,
                        top=0.93, wspace=0.05, hspace=0.15)

    df = pd.read_excel('/mnt/e/ERA5/每月天数.xlsx')
    mn_str = df['月份']

    t = np.arange(1981, 2021)
    t2 = np.arange(1982, 2019)

    ind = 0
    for i in range(4):
        for j in range(3):
            '''
            #渐变色柱状图
            #归一化
            norm = plt.Normalize(-2.3, 2.3) #值的范围
            norm_values = norm(data1[ind, :])
            map_vir = cm.get_cmap(name='bwr_r')
            colors = map_vir(norm_values)
            '''
            axs[i, j].plot(t2, data1[:, ind], color='k', label="GPP")
            axs[i, j].scatter(t2, data1[:, ind], color='k')
            ax2 = axs[i, j].twinx()
            ax2.bar(t2, data2[:, ind], color='orange',
                    label="GPP Anomaly", zorder=2, alpha=0.5)

            axs[i, j].text(1980.5, 140, f"{mn_str[ind]}", fontsize=15)
            ind += 1
            axs[i, j].set_xlim(1979, 2022)
            axs[i, j].set_ylim(-5, 155)
            axs[i, j].set_yticks(np.arange(0, 151, 30))
            axs[i, j].tick_params(labelsize=15)

            ax2.set_ylim(-30, 30)
            ax2.set_yticks(np.arange(-30, 31, 10))

            ax2.tick_params(axis='y', labelsize=15, colors="orange")
            ax2.spines["right"].set_color("orange")
            if j != 2:
                ax2.get_yaxis().set_visible(False)

    # 添加图例，本质上line1为list
    line1, label1 = axs[0, 0].get_legend_handles_labels()
    line2, label2 = ax2.get_legend_handles_labels()
    lines = line1+line2
    labels = label1+label2
    fig.legend(lines, labels, loc='upper right',
               bbox_to_anchor=(0.95, 1), fontsize=20)

    plt.suptitle(f"{title}", fontsize=25)
    plt.savefig(rf'/mnt/e/GLASS-GPP/JPG MG/{title}.jpg',
                bbox_inches='tight')
    plt.show()


# %%
def plot2(data1, data2, title):
    print(np.nanmax(data1), np.nanmin(data1))
    print(np.nanmax(data2), np.nanmin(data2))

    fig, axs = plt.subplots(4, 3, figsize=(
        20, 12), dpi=150, sharey=True, sharex=True)
    fig.subplots_adjust(left=0.05, bottom=0.1, right=0.95,
                        top=0.93, wspace=0.05, hspace=0.15)

    df = pd.read_excel('/mnt/e/ERA5/每月天数.xlsx')
    mn_str = df['月份']

    t = np.arange(1981, 2021)
    t2 = np.arange(1982, 2019)

    ind = 0
    for i in range(4):
        for j in range(3):
            '''
            #渐变色柱状图
            #归一化
            norm = plt.Normalize(-2.3, 2.3) #值的范围
            norm_values = norm(data1[ind, :])
            map_vir = cm.get_cmap(name='bwr_r')
            colors = map_vir(norm_values)
            '''
            axs[i, j].plot(t2, data1[:, ind], color='k', label="WUE")
            axs[i, j].scatter(t2, data1[:, ind], color='k')
            ax2 = axs[i, j].twinx()
            ax2.bar(t2, data2[:, ind], color='#7B68EE',
                    label="WUE Anomaly", zorder=2, alpha=0.5)

            axs[i, j].text(1980.5, 2.6, f"{mn_str[ind]}", fontsize=15)
            ind += 1
            axs[i, j].set_xlim(1979, 2022)
            axs[i, j].set_ylim(-1, 3)
            axs[i, j].set_yticks(np.arange(-1, 3.1, 1))
            axs[i, j].tick_params(labelsize=15)

            ax2.set_ylim(-1, 1)
            ax2.set_yticks(np.arange(-1, 1.1, 0.5))

            ax2.tick_params(axis='y', labelsize=15, colors="#7B68EE")
            ax2.spines["right"].set_color("#7B68EE")
            if j != 2:
                ax2.get_yaxis().set_visible(False)

    # 添加图例，本质上line1为list
    line1, label1 = axs[0, 0].get_legend_handles_labels()
    line2, label2 = ax2.get_legend_handles_labels()
    lines = line1+line2
    labels = label1+label2
    fig.legend(lines, labels, loc='upper right',
               bbox_to_anchor=(0.95, 1), fontsize=20)

    plt.suptitle(f"{title}", fontsize=25)
    plt.savefig(rf'/mnt/e/GLASS-GPP/JPG MG/{title}.jpg',
                bbox_inches='tight')
    plt.show()


# %%
# plot(gpp_MG, gpp_a_MG, f"MG Grassland GPP Month")
# plot2(wue_MG, wue_a_MG, f"MG Grassland WUE Month")
