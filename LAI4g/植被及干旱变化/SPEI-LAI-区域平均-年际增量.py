# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 13:35:42 2023

@author: MaYutong
"""
import cmaps

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

plt.rcParams['font.sans-serif']='times new roman'

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
    inpath = (r"E:/LAI4g/data_MG/LAI_82_20_MG_SPEI0.5x0.5.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        # print(f.variables['lai'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        lai = (f.variables['lai'][:, 4:9, :, :])
        lai_gs = np.nanmean(lai, axis=1)
        lai_diff = lai_gs[1:, ]-lai_gs[:-1, ]

    return lai_diff

# %%


def read_nc2():
    inpath = (r"E:/SPEI_base/data/spei03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        spei = (f.variables['spei'][972:, :, :])
        spei = spei.reshape(39, 12, 30, 50)
        spei_gsl = spei[:, 4:9, :]
        spei_gsl = np.nanmean(spei_gsl, axis=1)  # 82-20
        spei_gsl2 = spei_gsl[1:, ]  # 83-20
        spei_diff = spei_gsl2-spei_gsl[:-1, ]

    return spei_diff


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


def plot(data1, data2, title):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), dpi=500, sharex=True)

    fig.subplots_adjust(left=0.05, bottom=0.1, right=0.95,
                        top=0.9, wspace=None, hspace=0.2)

    t_date = pd.date_range(f'1983', periods=38, freq="YS")
    tt = []
    for j, x in enumerate(t_date):
        tt.append(x.strftime("%Y"))
    t2 = np.arange(38)

    for i in range(2):
        ax = axes[i]
        ax.axhline(y=0, c="k", linestyle="--")
        # ax.axvline(16, color="b", linewidth=2, zorder=3)
        ax.fill_between([14.5, 16.5], -30, 30, facecolor='dodgerblue', alpha=0.4, zorder=0)
        ##########################
        if i == 0:
            # ax.text(16, 1.7, "1999(drought turning point)", c='b', fontsize=20)
            # 渐变色柱状图
            # 归一化
            norm = plt.Normalize(-0.5, 0.5)  # 值的范围
            norm_values = norm(data1)
            # cmap = cmaps.BlueWhiteOrangeRed_r
            cmap = cmaps.MPL_bwr_r
            map_vir = cm.get_cmap(cmap)
            colors = map_vir(norm_values)

            ax.bar(t2, data1, color=colors, edgecolor='none',
                   label="SPEI03", zorder=1)
            ax.tick_params(labelsize=20)
            # ax.set_ylabel("SPEI03", fontsize=20)
            ax.set_ylim(-2, 2)

            ax.set_xticks(t2[::3])
            ax.set_xticklabels(tt[::3])
            ax.set_title("(a) SPEI03", loc="left", fontsize=20)
        ##########################
        elif i == 1:
            # ax.text(16, 0.25, "1999(drought turning point)", c='b', fontsize=20)
            ax = axes[i]
            # 渐变色柱状图
            # 归一化
            norm = plt.Normalize(-0.05, 0.05)  # 值的范围
            norm_values = norm(data2)
            cmap = cmaps.MPL_BrBG[20:109]
            map_vir = cm.get_cmap(cmap)
            colors = map_vir(norm_values)

            ax.bar(t2, data2, color=colors,
                   edgecolor='none', label="LAI", zorder=1)
            ax.tick_params(labelsize=20)
            # ax.set_ylabel("units: m2/m2", fontsize=20)
            ax.set_ylim(-0.25, 0.25)
            ax.set_title("(b) LAI(units: m${^2}$/m${^2}$)", loc="left", fontsize=20)
    # label
    # handles1, labels1 = ax.get_legend_handles_labels()
    # handles2, labels2 = ax2.get_legend_handles_labels()
    # ax.legend(handles1+handles2, labels1+labels2, ncol=2, fontsize=20, loc="upper left")

    # plt.suptitle(f'{title}', fontsize=20)
    plt.savefig(rf'E:/LAI4g/JPG_MG/{title}.jpg',
              bbox_inches='tight')
    plt.show()


# %%
lcc = read_lcc()
lai = read_nc()
lai_mk = mask(130, lai)
print("LAI", np.nanmax(lai_mk), np.nanmin(lai_mk))

spei = read_nc2()
spei_mk = mask(130, spei)
print("SPEI", np.nanmax(spei_mk), np.nanmin(spei_mk))

plot(spei_mk, lai_mk, title=r"Grassland LAI SPEI GSL Interannual increment")

# %%
year = np.arange(1983, 2021)
print("Wet->Dry:", year[spei_mk<-0.8], "\n")

print("Dry->Wet:", year[spei_mk>0.8], "\n")
