# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 20:04:29 2023

@author: MaYutong
"""

# %%
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
    global lat_MG, lon_MG, lat, lon
    inpath = r"E:/CRU/VAP_DATA/vpd_CRU_MONTH_81_20.nc"
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        lat = (f.variables['lat'][:]).data
        lon = (f.variables['lon'][:]).data
        vpd = (f.variables['vpd'][:])

        vpd_MG, lat_MG, lon_MG = vpd_xarray(vpd)
        vpd_MG_gsl = np.nanmean(vpd_MG[4:9, :], axis=0)
    return vpd_MG_gsl

# %% mask数组


def mask(x, data):
    lcc2 = np.array(lcc)

    np.place(lcc2, lcc2 != x, 1)
    np.place(lcc2, lcc2 == x, 0)

    spei_ma = np.empty((40, 30, 50))
    spei_ma = np.ma.array(spei_ma)

    for l in range(40):
        a = data[l, :, :]
        a = ma.masked_array(a, mask=lcc2, fill_value=-999)
        spei_ma[l, :, :] = a

    spei_ma_ave = np.nanmean(spei_ma, axis=(1, 2))

    return spei_ma_ave

# %%


def plot(data1, title):
    fig = plt.figure(figsize=(12, 4), dpi=500)

    t = np.arange(1981, 2021)

    # 滑动相关
    t2 = np.arange(1985, 2016)
    data1_roll = pd.Series(data1)
    data1_roll = data1_roll.rolling(10).mean().dropna()

    # 趋势线拟合
    z1 = np.polyfit(t, data1, 1)
    p1 = np.poly1d(z1)
    data1_pred = p1(t)

    ax = fig.add_axes([0.2, 0.1, 0.3, 0.4])
    ax.axvline(1999, color="k", linewidth=1, zorder=3)
    ax.text(1999, 2, "1999(turning point)", c='k')
    ax.plot(t, data1_pred, c="k", linestyle="--", linewidth=1, zorder=3)
    # 渐变色柱状图
    # 归一化
    norm = plt.Normalize(-0.2, 0.2)  # 值的范围
    norm_values = norm(data1)
    cmap = cmaps.MPL_bwr_r
    map_vir = cm.get_cmap(cmap)
    colors = map_vir(norm_values)

    ax.bar(t, data1, color=colors, label="SPEI03", zorder=2)
    ax.axhline(0, color="k", linewidth=0.3)
    # ax.plot(t2, data1_roll, c='b', zorder=3, linewidth=1)

    ax.set_title(f"Growing Season VPD", fontsize=10, loc="left")
    ax.set_ylim(-2.5, 2.5)
    ax.set_yticks(np.arange(-2, 2.1, 1))
    ax.tick_params(labelsize=10)

    # fig.suptitle(f"{title}", fontsize=20)
    plt.savefig(rf'E:/CRU/JPG_MG/{title}.jpg',
                bbox_inches='tight')
    plt.show()


# %%
lcc = read_lcc()
vpd = read_nc()
vpd_ave = mask(130, vpd)
vpd_anom = vpd_ave-vpd_ave.mean()
print(np.nanmax(vpd_anom), np.nanmin(vpd_anom))
plot(vpd_anom, f"MG Grassland VPD Anomaly Annual")
