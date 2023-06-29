# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 20:51:06 2023

@author: MaYutong
"""
import cmaps

import netCDF4 as nc
from matplotlib import cm
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


def read_nc():
    inpath = (r"E:/SPEI_base/data/spei03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        spei = (f.variables['spei'][960:, :, :])
        spei = spei.reshape(40, 12, 30, 50)
        spei_gsl = spei[:, 4:9, :].reshape(40*5, 30, 50)
    return spei_gsl


# %% mask数组
def mask(x, data):
    sp = data.shape[0]
    lcc2 = np.array(lcc)

    np.place(lcc2, lcc2 != x, 1)
    np.place(lcc2, lcc2 == x, 0)

    spei_ma = np.empty((sp, 30, 50))
    spei_ma = np.ma.array(spei_ma)

    for l in range(sp):
        a = data[l, :, :]
        a = np.ma.masked_array(a, mask=lcc2, fill_value=-999)
        spei_ma[l, :, :] = a

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
def plot(data1, title):
    fig = plt.figure(figsize=(16, 4), dpi=500)

    t_date = t_xaxis()
    t = np.arange(200)

    # 滑动相关
    # t2 = np.arange(1985, 2016)
    # data1_roll = pd.Series(data1)
    # data1_roll = data1_roll.rolling(10).mean().dropna()

    # 趋势线拟合
    z1 = np.polyfit(t, data1, 1)
    p1 = np.poly1d(z1)
    data1_pred = p1(t)

    ax = fig.add_axes([0.2, 0.1, 0.3, 0.4])
    ax.axvline(95, color="b", linewidth=1, zorder=3)
    ax.text(95, 2., "1999(turning point)", c='b')
    ax.plot(t, data1_pred, c="b", linestyle="--", linewidth=1, zorder=3)
    # 渐变色柱状图
    # 归一化
    norm = plt.Normalize(-2.5, 2.5)  # 值的范围
    norm_values = norm(data1)
    cmap = cmaps.BlueWhiteOrangeRed_r
    map_vir = cm.get_cmap(cmap)
    colors = map_vir(norm_values)

    ax.bar(t, data1, color=colors, label="SPEI03", zorder=2)
    ax.axhline(0, color="k", linewidth=0.3)
    # ax.plot(t2, data1_roll, c='b', zorder=3, linewidth=1)

    ax.set_title(f"1981-2020 GSL SPEI03", fontsize=10, loc="left")
    ax.set_ylim(-2.5, 2.5)
    ax.set_yticks(np.arange(-2, 2.1, 1))
    ax.tick_params(labelsize=10)
    ax.set_xlim(-10, 210)
    ax.set_xticks(t[::10])
    ax.set_xticklabels(t_date[::10], rotation=60)


    # fig.suptitle(f"{title}", fontsize=20)
    # plt.savefig(rf'E:/SPEI_base/python MG/JPG/{title}.jpg',
    #             bbox_inches='tight')
    # plt.savefig(rf'E:/LAI4g/JPG_MG/Rmax(Scale)/{title}.jpg',
    #             bbox_inches='tight')
    plt.show()


# %%
lcc = read_lcc()
spei = read_nc()
spei_mk = mask(130, spei)
print(np.nanmax(spei_mk), np.nanmin(spei_mk))
plot(spei_mk, title=r"Grassland 1981-2020 GSL SPEI03")
