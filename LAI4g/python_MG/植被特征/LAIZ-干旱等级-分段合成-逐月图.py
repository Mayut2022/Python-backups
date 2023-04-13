# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 09:26:24 2023

@author: MaYutong
"""
import cmaps
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.io.shapereader import Reader, natural_earth

from eofs.standard import Eof
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.ticker as mticker
from matplotlib.patches import Rectangle

import netCDF4 as nc

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import xarray as xr


plt.rcParams['font.sans-serif'] = ['times new roman']  # 指定默认字体


# %%


def read_lcc():
    inpath = (r"E:/ESA/ESA_LUC_2000_SPEI03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        # print(f.variables['lccs_class'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        lcc = (f.variables['lcc'][:])

    return lcc
# %%


def read_nc():
    global lat, lon
    inpath = (r"E:/LAI4g/data_MG/LAI_Zscore_82_20_MG_SPEI0.5x0.5.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        # print(f.variables['lai'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        lai = (f.variables['lai'][:, 4:9, :, :])
        lai1, lai2 = lai[:18, ], lai[18:, ]

        lai11 = lai1.reshape(18*5, 30, 50)
        lai22 = lai2.reshape(21*5, 30, 50)

    return lai11, lai22


def read_nc2():
    inpath = (r"E:/SPEI_base/data/spei03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        spei = (f.variables['spei'][972:, :, :])
        spei = spei.reshape(39, 12, 30, 50)
        spei_gsl = spei[:, 4:9, :].reshape(39*5, 30, 50)
        spei1, spei2 = spei_gsl[:90, :], spei_gsl[90:, :]
    return spei1, spei2


# %%


def t_xaxis():
    year = np.arange(1981, 2021)
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
# %% mask数组


def mask(x, data):
    sp = data.shape
    lcc2 = np.array(lcc)

    np.place(lcc2, lcc2 != x, 1)
    np.place(lcc2, lcc2 == x, 0)

    spei_ma = np.empty((sp[0], 30, 50))
    spei_ma = np.ma.array(spei_ma)

    for l in range(sp[0]):
        a = data[l, :, :]
        a = np.ma.masked_array(a, mask=lcc2)
        spei_ma[l, :, :] = a

    spei_ma_ave = np.nanmean(spei_ma, axis=(1, 2))

    return spei_ma_ave


# %% mask数组, 统计干旱月份和对应的数据
# 筛选干旱与非干旱


def lai_drought(x, lai, spei):
    print(str(x))

    # 非干旱统计
    # 植被值
    laim1 = lai.copy()
    exec(f"laim1[{x}]=np.nan")
    laim_N = locals()['laim1']
    gNm = mask(130, laim_N)
    # 月份统计
    laic1 = lai.copy()
    exec(f"laic1[{x}]=0")
    exec(f"laic1[~({x})]=1")
    laic_N = locals()['laic1']
    gNc =  mask(130, laic_N)

    # 干旱统计
    # 植被值
    laim2 = lai.copy()
    exec(f"laim2[~({x})]=np.nan")
    laim_Y = locals()['laim2']
    gYm = mask(130, laim_Y)
    # 月份统计
    laic2 = lai.copy()
    exec(f"laic2[~({x})]=0")
    exec(f"laic2[{x}]=1")
    laic_Y = locals()['laic2']
    gYc = mask(130, laic_Y)

    return gNm, gNc, gYm, gYc  # 依次为：非干旱的植被平均值，非干旱月数；干旱的植被平均值，干旱月数
# %%


def com_plot(data1, data2, title):
    global t_date
    print(np.nanmax(data1), np.nanmin(data1))
    print(np.nanmax(data2), np.nanmin(data2))

    fig = plt.figure(1, figsize=(18, 9), dpi=500)
    fig.subplots_adjust(left=0.05, bottom=0.1,
                        right=0.95, top=0.90, hspace=0.15)
    axes = fig.subplots(2, 1, sharex=True)

    t = np.arange(195)
    t_date = t_xaxis()[5:]

    # 趋势线拟合
    # z1 = np.polyfit(t, data2, 1)
    # p1 = np.poly1d(z1)
    # data2_pred = p1(t)
    # print(data2_pred)

    ###########
    ax = axes[0]
    ax.axvline(85, color="b", linewidth=2, zorder=2)
    ax.text(85, 0.92, "1999(turning point)", c='b', fontsize=15)
    ax.bar(t, data1, color="orange", width=0.8, alpha=0.8)
    ax.set_xlim(-10, 205)
    ax.set_xticks(t[::10])
    ax.set_xticklabels(t_date[::10], rotation=60)
    # ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y'))
    ax.set_ylim(0, 0.1)
    yticks = np.arange(0, 1.1, 0.2)
    ax.set_yticks(yticks)
    ylabels = [f"{100*x:.0f}%" for x in yticks]
    ax.set_yticklabels(ylabels)
    ax.tick_params(labelsize=15)
    ax.set_title("(a) Drought Area Percentage", fontsize=15, loc="left")

    ###########
    ax2 = axes[1]
    ax2.axvline(85, color="b", linewidth=2, zorder=2)
    ax2.text(85, 1.45, "1999(turning point)", c='b', fontsize=15)
    ax2.axhline(y=0, c="k", linestyle="-", linewidth=1)
    ax2.axhline(gYm1_ave, xmin=0, xmax=(95/215),
                c="k", linestyle="--", linewidth=2)
    ax2.axhline(gYm2_ave, xmin=(95/215), xmax=1,
                c="r", linestyle="--", linewidth=2)
    ax2.bar(t, data2, color='green', alpha=0.8)
    ax2.set_ylim(-2.4, 2.4)
    yticks = np.arange(-2, 2.1, 0.5)
    ax2.set_yticks(yticks)
    ax2.tick_params(labelsize=15)
    ax2.set_title("(b) LAI Z-score", fontsize=15, loc="left")
    # ax2.plot(t, data2_pred, c="b", linestyle="--", linewidth=1, zorder=3)

    # ax2.legend()

    plt.suptitle(f'{title}', fontsize=20)
    # plt.savefig(
    #     rf'E:/LAI4g/JPG_MG/Comparison/{title}.jpg', bbox_inches='tight')
    plt.show()

# %%


def PLOT(data1, data2, name):
    # data3 = data2-data1

    title = rf"{name} Drought LAIZ (SPEI03)"
    # title = rf"{name} Drought (SPEI12 01)"
    print("\n", title)

    # print('5 percentile is: ', np.nanpercentile(data1, 5))
    # print('95 percentile is: ', np.nanpercentile(data2, 95), "\n")

    # print("DIFF")
    # print('5 percentile is: ', np.nanpercentile(data3, 5))
    # print('95 percentile is: ', np.nanpercentile(data3, 95), "\n")

    com_plot(data1, data2, title)


# %%
lcc = read_lcc()
lai1, lai2 = read_nc()
spei1, spei2 = read_nc2()

# %%
lev_d = [
    "(spei<-0.5)&(spei>=-1)", "(spei<-1)&(spei>=-1.5)", "(spei<-1.5)&(spei>=-2)", "(spei<-2)"]
lev_n = ["Mild", "Moderate", "Severe", "Extreme"]

for lev1, name in zip(lev_d[:], lev_n[:]):
    _, _, gYm1, gYc1 = lai_drought(lev1, lai1, spei1)
    _, _, gYm2, gYc2 = lai_drought(lev1, lai2, spei2)
    gYm1_ave, gYm2_ave = np.nanmean(gYm1), np.nanmean(gYm2)
    gYm = np.hstack((gYm1, gYm2))
    gYc = np.hstack((gYc1, gYc2))
    PLOT(gYc, gYm, name)
