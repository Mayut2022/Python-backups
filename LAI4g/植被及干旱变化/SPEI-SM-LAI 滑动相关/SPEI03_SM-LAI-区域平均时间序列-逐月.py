# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 12:39:01 2023

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

# plt.rcParams['font.sans-serif'] = ['times new roman']  # 指定默认字体
plt.rcParams['font.sans-serif']=['simsun']
plt.rcParams['axes.unicode_minus'] = False

import warnings
warnings.filterwarnings("ignore")

# %%


def read_nc(inpath):
    global lat, lon
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        r1 = (f.variables['corr1'][:])
        r2 = (f.variables['corr2'][:])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
    return r1, r2


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

def plot_Grass(data1, win, sig, title):
    
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
    ax.axhline(y=sig, c="r", linestyle="--")
    ax.axvline(85, color="b", linewidth=2, zorder=2)
    ax.text(85, 0.58, "1999(turning point)", c='b', fontsize=15)
    # "#ffb432"Grass;  "#285000" Forest
    ax.scatter(t[wh:], data1[w:], c='k', s=10)
    ax.plot(t[wh:], data1[w:], c='k', label="Grassland", linewidth=1)
    ax.tick_params(labelsize=15)
    ax.set_ylabel("Correlation coefficient", fontsize=15)
    ax.set_ylim(-0.3, 0.65)
    ax.set_yticks(np.arange(-0.3, 0.61, 0.15))
    ax.set_xlim(-10, 205)
    ax.set_xticks(t[::5])
    ax.set_xticklabels(t_date[::5], rotation=60)
    # ax.xaxis.set_major_formatter(mdate.DateFormatter('%y-%m'))

    plt.legend(loc='upper right', fontsize=15)
    plt.xlabel("years", fontsize=15)
    plt.suptitle(f'{title}', fontsize=20)
    plt.savefig(rf'E:/LAI4g/JPG_MG2/{title}.jpg',
                bbox_inches='tight')
    plt.show()





# %%
lcc = read_lcc()
win_all = np.arange(10, 51, 10)
sig_all = [0.5494, 0.3783, 0.3061, 0.2638, 0.2353]
# win = 30
for win, sig in zip(win_all[2:], sig_all[2:]):
    inpath = rf"E:/LAI4g/data_MG/moving corr/LAI_SPEI_SM_Partial_detrend_Win{win}.nc"
    r1, r2 = read_nc(inpath)
    r1_g, r2_g = mask(130, r1), mask(130, r2)
    print("草地 LAI-SPEI(Partial SM)", np.nanmax(r1_g), np.nanmin(r1_g))
    print("草地 LAI-SM(Partial SPEI)", np.nanmax(r2_g), np.nanmin(r2_g))
    plot_Grass(r1_g, win, sig, title=rf"偏相关 LAI to SPEI03 (Partial 同期SM Win={win}) ")
    plot_Grass(r2_g, win, sig, title=rf"偏相关 LAI to 同期SM (Partial SPEI03 Win={win}) ")
