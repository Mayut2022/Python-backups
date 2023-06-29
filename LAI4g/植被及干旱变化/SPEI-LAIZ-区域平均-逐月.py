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
from scipy.stats import pearsonr
from sklearn import preprocessing
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
    # inpath = (r"E:/LAI4g/data_MG/LAI_Detrend_Zscore_82_20_MG_SPEI0.5x0.5.nc")
    inpath = (r"E:/LAI4g/data_MG/LAI_Zscore_82_20_MG_SPEI0.5x0.5.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        # print(f.variables['lai'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        lai = (f.variables['lai'][:, 4:9, :, :])
        lai_gs = lai.reshape(195, 30, 50)

    return lai_gs

# %%


def read_nc2():
    inpath = (r"E:/SPEI_base/data/spei03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        spei = (f.variables['spei'][960:, :, :])
        spei = spei.reshape(40, 12, 30, 50)
        spei_gsl = spei[:, 4:9, :].reshape(40*5, 30, 50)
    return spei_gsl

# def read_nc2():
#     inpath = (r"E:/SPEI_base/data/spei03_MG_detrend.nc")
#     with nc.Dataset(inpath) as f:
#         # print(f.variables.keys())
#         spei = (f.variables['spei'][:])
#         spei_gsl = spei[4:9, :].reshape(40*5, 30, 50)
#     return spei_gsl

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

# %%


# def plot(data1, title):
#     fig = plt.figure(1, figsize=(16, 4), dpi=500)

#     fig.subplots_adjust(left=0.05, bottom=0.1, right=0.95,
#                         top=0.9, wspace=None, hspace=0.12)

#     t = np.arange(195)
#     t_date = t_xaxis()[5:]
#     ax = fig.subplots(1)
#     ax.axhline(y=0, c="k", linestyle="--")

#     ax.axvline(90, color="b", linewidth=1, zorder=3)
#     ax.text(90, 1.7, "1999(drought turning point)", c='b', fontsize=15)
#     ax.scatter(t[:], data1, c='k', s=20)
#     ax.plot(t[:], data1, c='k', label="LAI", linewidth=2)
#     ax.tick_params(labelsize=15)
#     ax.set_xlim(-10, 205)
#     ax.set_xticks(t[::10])
#     ax.set_xticklabels(t_date[::10], rotation=60)
#     ax.set_ylim(-2., 2.)
#     ax.set_yticks(np.arange(-2, 2.1, 1))
#     ax.set_ylabel("units: m2/m2", fontsize=15)

#     plt.legend(loc='upper left', fontsize=15)
#     plt.xlabel("years", fontsize=15)

#     ax.set_title(f"1982-2020 GSL LAI", fontsize=15, loc="left")

#     # plt.suptitle(f'{title}', fontsize=20)
#     plt.savefig(rf'E:/LAI4g/JPG_MG/{title}.jpg',
#                 bbox_inches='tight')
#     plt.show()


# %%

def plot_all(data1, data2, title):
    fig = plt.figure(1, figsize=(16, 4), dpi=500)

    fig.subplots_adjust(left=0.05, bottom=0.1, right=0.95,
                        top=0.9, wspace=None, hspace=0.12)

    t = np.arange(200)
    t_date = t_xaxis()[:]
    ax = fig.subplots(1)
    ax.axhline(y=0, c="k", linestyle="--")

    ax.axvline(90, color="b", linewidth=1, zorder=3)
    ax.text(90, -2.2, "1999(drought turning point)", c='b', fontsize=15)
    ax.text(100, 2, "CORR: 0.36", c='r', fontsize=25, weight="bold")
    ax.scatter(t[5:], data1, c='k', s=10, zorder=2)
    ax.plot(t[5:], data1, c='k', label="LAI", linewidth=1.5, zorder=2)
    
    # 渐变色柱状图
    # 归一化
    norm = plt.Normalize(-2.5, 2.5)  # 值的范围
    norm_values = norm(data2)
    cmap = cmaps.BlueWhiteOrangeRed_r
    map_vir = cm.get_cmap(cmap)
    colors = map_vir(norm_values)
    
    ax.bar(t, data2, color=colors, label="SPEI03", zorder=1)
    
    
    ax.tick_params(labelsize=15)
    ax.set_xlim(-10, 210)
    ax.set_xticks(t[::10])
    ax.set_xticklabels(t_date[::10], rotation=60)
    ax.set_ylim(-2.5, 2.5)
    ax.set_yticks(np.arange(-2, 2.1, 1))
    
    # ax.set_ylabel("units: m2/m2", fontsize=15)

    plt.legend(loc='upper left', fontsize=15)
    plt.xlabel("years", fontsize=15)

    ax.set_title(f"1982-2020 GSL LAI&SPEI Detrend", fontsize=15, loc="left")

    # plt.suptitle(f'{title}', fontsize=20)
    # plt.savefig(rf'E:/LAI4g/JPG_MG/{title}.jpg',
    #             bbox_inches='tight')
    plt.show()




# %%
lcc = read_lcc()
lai = read_nc()
lai_mk = mask(130, lai)
print(np.nanmax(lai_mk), np.nanmin(lai_mk))
spei = read_nc2()
spei_mk = mask(130, spei)
print(np.nanmax(spei_mk), np.nanmin(spei_mk))
# plot(lai_mk, title=r"MG Grassland LAI GSL Month 82-20")
# plot_all(lai_mk, spei_mk, title=r"Grassland LAI&SPEI GSL Month Detrend")

# %%


# def trend(data, var):
#     t = np.arange(len(data))
#     s, _, _, p, _ = linregress(t, data)

#     print(var)
#     print("slope:", s, "p-value:", p, "\n")


# # %% 82-00
# trend(lai_mk[:19], "LAI 82-99")

# # %% 00-18
# trend(lai_mk[19:], "LAI 00-18")

#%%
# t = t_xaxis()[5:]
# t[90]



# %% 计算时间序列相关性
spei_mk2 = spei_mk[5:]
pccs = pearsonr(lai_mk, spei_mk2)
print(pccs)

# print("去趋势后的相关系数：", "0.01")

print("原场的相关系数：", "0.36")
print("转折前的相关系数：", "0.36")
print("转折后的相关系数：", "0.47")