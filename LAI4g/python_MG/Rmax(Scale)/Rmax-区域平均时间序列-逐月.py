# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 19:15:52 2023

@author: MaYutong
"""

from collections import Counter
import cmaps
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.io.shapereader import Reader, natural_earth

from eofs.standard import Eof
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.ticker as mticker
from matplotlib.patches import Rectangle

import netCDF4 as nc

from matplotlib import cm
import numpy as np
import numpy.ma as ma
import pandas as pd

plt.rcParams['font.sans-serif'] = ['times new roman']  # 指定默认字体

# %%


def read_nc(inpath):
    global lat, lon
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())

        r = (f.variables['Rmax'][:])
        r_s = (f.variables['Rmax_scale'][:])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])

        r_c = r.copy()
        r_c[r_c >= 0] = 1
        r_c[r_c < 0] = -1

    return r, r_s, r_c


# %%
def read_lcc():
    inpath = (r"E:\ESA\ESA_LUC_2000_SPEI03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        # print(f.variables['lccs_class'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        lcc = (f.variables['lcc'][:])

    return lcc


# %% mask数组


def mask(x, data):
    sp = data.shape[0]
    lcc2 = np.array(lcc)

    np.place(lcc2, lcc2 != x, 1)
    np.place(lcc2, lcc2 == x, 0)

    spei_ma = np.empty((sp, 30, 50))
    spei_ma = np.ma.array(spei_ma)

    for i in range(sp):
        a = data[i, :, :]
        a = ma.masked_array(a, mask=lcc2)
        spei_ma[i, :, :] = a

    spei_ma_ave = np.nanmean(spei_ma, axis=(1, 2))

    return spei_ma_ave

# %%


def t_xaxis():
    year = np.arange(1982, 2021)
    for i, yr in enumerate(year):
        t = pd.date_range(f'{yr}/05', periods=5, freq="MS")
        tt = []
        for j, x in enumerate(t):
            tt.append(x.strftime("%Y-%m"))
        if i == 0:
            tt_all = tt
        else:
            tt_all = tt_all+tt

    return tt_all
# %%


def plot(data1, data2, title):

    _ = [np.nan]*15
    _ = np.array(_)
    data1, data2 = np.hstack((data1, _)), np.hstack((data2, _))

    fig = plt.figure(1, figsize=(16, 4), dpi=500)

    fig.subplots_adjust(left=0.05, bottom=0.1, right=0.95,
                        top=0.9, wspace=None, hspace=0.12)

    t_date = t_xaxis()
    t = np.arange(195)
    ax = fig.subplots(1)
    ax.axhline(y=0, c="k", linestyle="--")

    # "#ffb432"Grass;  "#285000" Forest
    ax.scatter(t[15:], data1[30:], c='#ffb432', s=10)
    ax.plot(t[15:], data1[30:], c='#ffb432', label="Grassland", linewidth=1)
    ax.scatter(t[15:], data2[30:], c='#285000', s=10)
    ax.plot(t[15:], data2[30:], c='#285000', label="Forest", linewidth=1)
    ax.tick_params(labelsize=15)
    ax.set_ylabel("Sensitivity Rmax", fontsize=15)
    ax.set_ylim(-0.7, 0.7)
    ax.set_yticks(np.arange(-0.6, 0.61, 0.2))
    ax.set_xlim(-10, 205)
    ax.set_xticks(t[::5])
    ax.set_xticklabels(t_date[::5], rotation=60)
    # ax.xaxis.set_major_formatter(mdate.DateFormatter('%y-%m'))

    plt.legend(loc='lower right', fontsize=15)
    plt.xlabel("years", fontsize=15)
    plt.suptitle(f'{title}', fontsize=20)
    plt.savefig(rf'E:/LAI4g/JPG_MG/Rmax(Scale)/{title}.jpg',
                bbox_inches='tight')
    plt.show()


# %%

def plot_Grass(data1, win, title):
    
    wh, w = int(win/2), win.copy() ####1/2滑动窗口，滑动窗口
    _ = [np.nan]*wh
    _ = np.array(_)
    data1= np.hstack((data1, _))

    fig = plt.figure(1, figsize=(16, 4), dpi=500)

    fig.subplots_adjust(left=0.05, bottom=0.1, right=0.95,
                        top=0.9, wspace=None, hspace=0.12)

    t_date = t_xaxis()
    t = np.arange(195)
    ax = fig.subplots(1)
    ax.axhline(y=0, c="k", linestyle="--")
    ax.axvline(85, color="b", linewidth=2, zorder=2)
    ax.text(85, 0.72, "1999(turning point)", c='b', fontsize=15)
    # "#ffb432"Grass;  "#285000" Forest
    ax.scatter(t[wh:], data1[w:], c='#ffb432', s=10)
    ax.plot(t[wh:], data1[w:], c='#ffb432', label="Grassland", linewidth=1)
    ax.tick_params(labelsize=15)
    ax.set_ylabel("Correlation Rmax", fontsize=15)
    ax.set_ylim(0, 0.8)
    ax.set_yticks(np.arange(0, 0.81, 0.2))
    ax.set_xlim(-10, 205)
    ax.set_xticks(t[::5])
    ax.set_xticklabels(t_date[::5], rotation=60)
    # ax.xaxis.set_major_formatter(mdate.DateFormatter('%y-%m'))

    plt.legend(loc='upper right', fontsize=15)
    plt.xlabel("years", fontsize=15)
    plt.suptitle(f'{title}', fontsize=20)
    plt.savefig(rf'E:/LAI4g/JPG_MG/Rmax(Scale)/{title}.jpg',
                bbox_inches='tight')
    plt.show()





# %%
lcc = read_lcc()
win_all = np.arange(10, 51, 10)
# win = 30
for win in win_all:
    inpath = rf"E:/LAI4g/data_MG/Rmax(Scale)/LAI_SPEI_MG_MonthWindows{win}.nc"
    r, _, _ = read_nc(inpath)
    r_g = mask(130, r)
    r_f = mask(80, r)
    print("草地", np.nanmax(r_g), np.nanmin(r_g))
    # print("森林", np.nanmax(r_f), np.nanmin(r_f))
    # plot(r_g, r_f, title=r"Sensitivity Rmax (Win=30)")
    plot_Grass(r_g, win, title=rf"Grassland Sensitivity LAI to SPEI (Win={win}) ")
