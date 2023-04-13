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

    return lai_gs

# %%


def read_nc2():
    inpath = (r"E:/SPEI_base/data/spei03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        spei = (f.variables['spei'][960:, :, :])
        spei = spei.reshape(40, 12, 30, 50)
        spei_gsl = spei[:, 4:9, :]
        spei_gsl = np.nanmean(spei_gsl, axis=1)
    return spei_gsl


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
# def plot(data1, title):
#     fig = plt.figure(1, figsize=(10, 4), dpi=500)

#     fig.subplots_adjust(left=0.05, bottom=0.1, right=0.95,
#                         top=0.9, wspace=None, hspace=0.12)

#     t = pd.date_range(f'1982', periods=39, freq="YS")
#     ax = fig.subplots(1)
#     # ax.axhline(y=0, c="k", linestyle="--")

#     ax.scatter(t[:], data1, c='k', s=20)
#     ax.plot(t[:], data1, c='k', label="LAI(m2/m2)", linewidth=2)
#     ax.tick_params(labelsize=15)
#     ax.set_ylabel("units: m2/m2", fontsize=15)

#     plt.legend(loc='upper left', fontsize=15)
#     plt.xlabel("years", fontsize=15)
#     plt.suptitle(f'{title}', fontsize=20)
#     # plt.savefig(rf'E:/LAI4g/JPG_MG/{title}.jpg',
#     #           bbox_inches='tight')
#     plt.show()


# %%
def plot_all(data1, data2, title):
    global handles1, labels1, handles2, labels2
    fig = plt.figure(1, figsize=(10, 4), dpi=500)

    fig.subplots_adjust(left=0.05, bottom=0.1, right=0.95,
                        top=0.9, wspace=None, hspace=0.12)

    t_date = pd.date_range(f'1981', periods=40, freq="YS")
    tt = []
    for j, x in enumerate(t_date):
        tt.append(x.strftime("%Y"))
    t2 = np.arange(40)
    ax = fig.subplots(1)
    
    ##########################
    
    ax.axhline(y=0, c="k", linestyle="--")
    ax.axvline(18, color="b", linewidth=2, zorder=3)
    ax.text(18, 1.7, "1999(drought turning point)", c='b', fontsize=15)
    # 渐变色柱状图
    # 归一化
    norm = plt.Normalize(-2, 2)  # 值的范围
    norm_values = norm(data2)
    cmap = cmaps.BlueWhiteOrangeRed_r
    map_vir = cm.get_cmap(cmap)
    colors = map_vir(norm_values)
    
    ax.bar(t2, data2, color=colors, edgecolor='lightgray', label="SPEI03", zorder=1)
    ax.tick_params(labelsize=15)
    ax.set_ylabel("SPEI03", fontsize=15)
    ax.set_ylim(-2, 2)
    
    ax.set_xticks(t2[::6])
    ax.set_xticklabels(tt[::6])
    
    ##########################
    ax2 = ax.twinx()
    ax2.scatter(t2[1:], data1, c='k', s=20, zorder=2)
    ax2.plot(t2[1:], data1, c='k', label="LAI(m2/m2)", linewidth=2, zorder=2)
    ax2.tick_params(labelsize=15)
    ax2.set_ylabel("units: m2/m2", fontsize=15)
    ax2.set_ylim(0.9, 1.5)
    
    ################## label
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1+handles2, labels1+labels2, ncol=2, fontsize=15, loc="upper left")
    
    
    plt.suptitle(f'{title}', fontsize=20)
    plt.savefig(rf'E:/LAI4g/JPG_MG/{title}.jpg',
              bbox_inches='tight')
    plt.show()


# %%
lcc = read_lcc()
lai = read_nc()
lai_mk = mask(130, lai)
print(np.nanmax(lai_mk), np.nanmin(lai_mk))

spei = read_nc2()
spei_mk = mask(130, spei)
print(np.nanmax(spei_mk), np.nanmin(spei_mk))

# plot(lai_mk, title=r"MG Grassland LAI GSL Mean 82-20")
plot_all(lai_mk, spei_mk, title=r"Grassland LAI&SPEI GSL Annual")

# %%


# def trend(data, var):
#     t = np.arange(len(data))
#     s, _, _, p, _ = linregress(t, data)

#     print(var)
#     print("slope:", s, "p-value:", p, "\n")


# # %% 82-00
# trend(lai_mk[:19], "GPP 82-99")

# # %% 00-18
# trend(lai_mk[19:], "GPP 00-18")
