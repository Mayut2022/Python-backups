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


def vpd_xarray(band1):
    t = np.arange(40)
    mn = np.arange(1, 13, 1)
    vpd = xr.DataArray(
        band1, dims=['mn', 't', 'y', 'x'], coords=[mn, t, lat, lon])
    vpd_MG = vpd.loc[:, :, 40:55, 100:125]
    lat_MG = vpd_MG.y
    lon_MG = vpd_MG.x
    vpd_MG = np.array(vpd_MG)
    return vpd_MG, lat_MG, lon_MG


def read_nc():
    global lat_MG, lon_MG, lat, lon, vpd_MG
    inpath = r"E:/CRU/VAP_DATA/vpd_anomaly_CRU_MONTH_81_20.nc"
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        lat = (f.variables['lat'][:]).data
        lon = (f.variables['lon'][:]).data
        vpd = (f.variables['vpd'][:])

        vpd_MG, lat_MG, lon_MG = vpd_xarray(vpd)
        vpd_MG_gsl = vpd_MG[4:9, :].reshape(200, 30, 50, order='F')
    return vpd_MG_gsl


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
    ax.axvline(95, color="k", linewidth=1, zorder=3)
    ax.text(95, 3.2, "1999(turning point)", c='k')
    ax.plot(t, data1_pred, c="k", linestyle="--", linewidth=1, zorder=3)
    # 渐变色柱状图
    # 归一化
    norm = plt.Normalize(-0.1, 0.1)  # 值的范围
    norm_values = norm(data1)
    cmap = cmaps.MPL_bwr_r
    map_vir = cm.get_cmap(cmap)
    colors = map_vir(norm_values)

    ax.bar(t, data1, color=colors, label="VPD", zorder=2)
    ax.axhline(0, color="k", linewidth=0.3)
    # ax.plot(t2, data1_roll, c='b', zorder=3, linewidth=1)

    ax.set_title(f"1981-2020 GSL VPD", fontsize=10, loc="left")
    ax.set_ylim(-4, 4)
    ax.set_yticks(np.arange(-4., 4.1, 2))
    ax.tick_params(labelsize=10)
    ax.set_xlim(-10, 210)
    ax.set_xticks(t[::10])
    ax.set_xticklabels(t_date[::10], rotation=60)

    # fig.suptitle(f"{title}", fontsize=20)
    plt.savefig(rf'E:/CRU/JPG_MG/{title}.jpg',bbox_inches='tight')
    plt.show()


# %%
lcc = read_lcc()
vpd = read_nc()
vpd_mk = mask(130, vpd)
print(np.nanmax(vpd_mk), np.nanmin(vpd_mk))
plot(vpd_mk, title=r"MG Grassland VPD Anomaly")
