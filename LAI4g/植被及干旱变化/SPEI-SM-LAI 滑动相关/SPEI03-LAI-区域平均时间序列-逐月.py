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
from scipy.stats import pearsonr

plt.rcParams['font.sans-serif'] = ['times new roman']  # 指定默认字体
# plt.rcParams['font.sans-serif']=['simsun']
# plt.rcParams['axes.unicode_minus'] = False

import warnings
warnings.filterwarnings("ignore")


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

lcc = read_lcc()

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


def read_nc(inpath):
    global lat, lon
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        r = (f.variables['corr'][:])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
    return r



def read_nc2(win):
    inpath = (r"E:/SPEI_base/data/spei03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        spei = (f.variables['spei'][972:, :, :])
        spei = spei.reshape(39, 12, 30, 50)
        spei_gsl = spei[:, 4:9, :].reshape(39*5, 30, 50)
        spei_g = mask(130, spei_gsl)
        spei_g_roll = pd.Series(spei_g)
        spei_g_roll = spei_g_roll.rolling(win).mean()
    return spei_g_roll



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

def plot_Grass(data1, data2, win, sig, corr, title):
    
    wh, w = int(win/2), win.copy() ####1/2滑动窗口，滑动窗口
    _ = [np.nan]*wh
    _ = np.array(_)
    data1= np.hstack((data1, _))
    data2= np.hstack((data2, _))

    fig = plt.figure(1, figsize=(12, 4), dpi=500)

    fig.subplots_adjust(left=0.05, bottom=0.1, right=0.95,
                        top=0.9, wspace=None, hspace=0.12)

    t_date = t_xaxis()
    t = np.arange(195)
    ax = fig.subplots(1)
    
    ax.text(90, 0.6, f"CORR:{corr:.2f}", c='k', fontsize=15, weight="bold")
    # ax.axvline(85, color="gray", linewidth=2, zorder=2)
    # ax.text(85, 0.6, "1999(turning point)", c='gray', fontsize=15, weight="bold")
    ax.text(-4, 0.6, "(d)", c='k', fontsize=15)
    # "#ffb432"Grass;  "#285000" Forest
    # ax.scatter(t[wh:], data1[w:], c='k', s=10)
    ax.plot(t[wh:], data1[w:], c='k', label="Moving Corr", linewidth=2)
    
    ax.tick_params(labelsize=15)
    ax.set_ylabel("Correlation coefficient", fontsize=15)
    ax.set_ylim(0.1, 0.65)
    ax.set_yticks(np.arange(0.15, 0.61, 0.15))
    ax.set_xlim(-5, 20)
    ax.set_xticks(t[::10])
    ax.set_xticklabels(t_date[::10], rotation=60)
    # ax.xaxis.set_major_formatter(mdate.DateFormatter('%y-%m'))
    
    #############
    ax2 = ax.twinx()
    # ax2.scatter(t[wh:], data2[w:], c='r', s=10)
    ax2.plot(t[wh:], data2[w:], color='b', linestyle="--", label="SPEI", linewidth=2)
    ax2.tick_params(labelsize=15)
    ax2.set_ylim(-1, 1)
    ax2.set_yticks(np.arange(-1, 1.1, 0.5))
    ax2.set_ylabel("SPEI", fontsize=15)
    
    ax.axhline(y=sig, c="r", linestyle="--", label="90% Significance level", linewidth=2)
    # ax.text(165, sig-0.03, "Significance level", c='r', fontsize=15, weight="bold")

    ################## label
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1+handles2, labels1+labels2, ncol=2, fontsize=12, loc="upper right")

    
    plt.xlabel("years", fontsize=15)
    # plt.suptitle(f'{title}', fontsize=20)
    plt.savefig(rf'E:/LAI4g/JPG_MG2/{title}.jpg',
                bbox_inches='tight')
    plt.show()





# %%
lcc = read_lcc()
win_all = np.arange(10, 51, 10)
sig_all = [0.5494, 0.3783, 0.3061, 0.2638, 0.2353] ####90%显著性水平
sig_all = [0.5494, 0.3783, 0.3061, 0.2638, 0.2353] ####90%显著性水平
# win = 30
for win, sig in zip(win_all[2:], sig_all[2:]):
    inpath = rf"E:/LAI4g/data_MG/moving corr/LAI_SPEI_detrend_Win{win}.nc"
    r = read_nc(inpath)
    r_g = mask(130, r)
    spei = read_nc2(win)
    print("草地", np.nanmax(r_g), np.nanmin(r_g))
   
    pccs = pearsonr(r_g[~np.isnan(r_g)], spei.dropna())
    print(pccs)
    plot_Grass(r_g, spei, win, sig, pccs[0], title=rf"相关 LAI-SPEI03 (Win={win}) ")

# %% 计算时间序列相关性

# pccs = pearsonr(r_g[~np.isnan(r_g)], spei.dropna())
# print(pccs)
# print("相关系数：", "-0.45")