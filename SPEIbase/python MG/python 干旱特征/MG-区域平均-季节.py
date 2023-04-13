# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 14:37:25 2022

@author: MaYutong
"""

import datetime as dt
import netCDF4 as nc
from matplotlib import font_manager
import matplotlib.dates as mdate
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma

import pandas as pd
from scipy.stats import linregress
import xarray as xr
plt.rcParams['font.family'] = 'Times New Roman'
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


def read_nc2(inpath):
    global spei, t

    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())

        spei = (f.variables['spei'][960:])
        time = (f.variables['time'][960:])
        t = nc.num2date(time, 'days since 1900-01-01 00:00:0.0').data


# %%
'''
a = spei[0, :, :]
import matplotlib.pyplot as plt

plt.figure(1, dpi=500)
plt.imshow(lcc, cmap='rainbow_r')
plt.colorbar(shrink=0.75)
plt.show()
'''
# %% mask数组


def mask(x):
    lcc2 = np.array(lcc)

    np.place(lcc2, lcc2 != x, 1)
    np.place(lcc2, lcc2 == x, 0)

    spei_ma = np.empty((480, 30, 50))
    spei_ma = np.ma.array(spei_ma)

    for i in range(480):
        a = spei[i, :, :]
        a = ma.masked_array(a, mask=lcc2)
        spei_ma[i, :, :] = a

    spei_ma_ave = spei_ma.mean(axis=(1, 2))

    return spei_ma_ave

# %%


def subplot(a, ax, std, x_ticks):
    im = ax.imshow(a, cmap="RdBu", vmin=-1.8, vmax=1.8)

    '''
    aa = a.copy()
    aa[np.logical_and(aa<std, aa>-std)] = np.nan
    
    im = ax.imshow(aa, cmap="RdBu", vmin=-2, vmax=2)
    '''
    y_ticks = ["Spring", "Summer", "Autumn", "Winter"]

    ax.set_xticks(np.arange(a.shape[1]))
    ax.set_yticks(np.arange(a.shape[0]))
    ax.set_xticklabels(x_ticks)
    ax.set_yticklabels(y_ticks, rotation=0)

    #ax.set_ylabel("seasons", fontsize=10)
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True, labelsize=9)

    for edge, spine in ax.spines.items():  # 边框
        spine.set_visible(True)

    # plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
        # rotation_mode="anchor") #坐标旋转

    '''major minor 刻度'''
    ax.set_xticks(np.arange(a.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(a.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False, length=1)
    ax.tick_params(which="major", bottom=1, left=1, length=2)

    textcolors = ("black", "yellow")
    kw = dict(horizontalalignment="center",
              verticalalignment="center",
              fontsize="10", weight="bold")

    fmt = "{x:.2f}"
    fmt = matplotlib.ticker.StrMethodFormatter(fmt)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            kw.update(color=textcolors[a[i, j] < -std or a[i, j] > std])
            text = im.axes.text(j, i, fmt(a[i, j], None), **kw)

    return im


def heatmap(a, b, std, title):
    fig = plt.figure(figsize=(20, 6), dpi=500)

    fig.subplots_adjust(left=0.05, bottom=0.1, right=0.95,
                        top=0.92, wspace=None, hspace=0.15)
    axs = fig.subplots(2, 1)

    axs[0] = subplot(a, axs[0], std, x_ticks=np.arange(1981, 2001, 1))
    axs[1] = subplot(b, axs[1], std, x_ticks=np.arange(2001, 2021, 1))

    '''
    cbar_ax = fig.add_axes([0.6, 0.25, 0.005, 0.5])
    cb = fig.colorbar(axs[0], cax=cbar_ax, orientation='vertical')
    '''

    plt.xlabel("SPEI03", fontsize=11)
    plt.xlabel("YEARS", fontsize=11)
    #plt.text(15, 13, f"threshold: SPEI<-{std:.2f}", color='dimgray', fontsize=10)
    plt.suptitle(f'{title}', fontsize=15)
    plt.savefig(
        rf'E:/SPEI_base/python MG/JPG/{title}.jpg', bbox_inches='tight')
    plt.show()


# %%
read_nc()
# lcc2 = np.flip(lcc, axis=0) lcc和SPEI都是反的，不需要翻转
# %%
'''
def read_plot(scale):
    inpath = (rf"E:/SPEI_base/data/{str(scale)}_MG.nc")
    read_nc2(inpath)
    spei_ave = mask(130)
    print(spei_ave.max(), spei_ave.min(), spei_ave.std())
    heatmap(spei_ave[:240], spei_ave[240:], 
            spei_ave.std(), f"Grassland {str(scale)} 81-20 heatmap")

scale = ["SPEI03", "SPEI06", "SPEI12"]
for sca in scale[:1]:
    read_plot(sca)
'''

# %% 480(输入data) -> 月 年


def mn_yr(data):
    tmp_mn = []
    for mn in range(12):
        tmp_ = []
        for yr in range(40):
            tmp_.append(data[mn])
            mn += 12
        tmp_mn.append(tmp_)

    tmp_mn = np.array(tmp_mn)

    return tmp_mn
# %%


def season_yr(data):
    tmp_s = np.vstack((data[2:, :], data[:2, :]))
    tmp_sea = []
    for mn1, mn2 in zip(range(0, 12, 3), range(3, 15, 3)):
        tmp_sea.append(tmp_s[mn1:mn2, :])

    tmp_sea = np.array(tmp_sea)
    tmp_sea = tmp_sea.mean(axis=1)

    return tmp_sea


# %%
inpath = (rf"E:/SPEI_base/data/SPEI03_MG.nc")
read_nc2(inpath)
spei_ave = mask(130)
spei_ave_mn = mn_yr(spei_ave)
spei_sea = season_yr(spei_ave_mn)
print(spei_sea.max(), spei_sea.min(), spei_sea.std())
# heatmap(spei_sea[:, :20], spei_sea[:, 20:],
#         spei_sea.std(), f"Grassland Season 81-20 heatmap")

# %%
def trend(data, var):
    t = np.arange(len(data))
    s, _, _, p, _ = linregress(t, data)

    print(var)
    print("slope:", s, "p-value:", p, "\n")
    
trend(spei_sea[1, :20], "SPEI 81-00")
trend(spei_sea[1, 20:], "SPEI 01-20")

trend(spei_sea[1, :], "SPEI 81-20")