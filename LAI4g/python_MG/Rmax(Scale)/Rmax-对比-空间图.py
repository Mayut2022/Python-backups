# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 21:18:59 2023

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
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.ticker as mticker
from matplotlib.patches import Rectangle

import netCDF4 as nc
import matplotlib.dates as mdate
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
import xarray as xr
import xlsxwriter


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
    # global a1, a2, o1, o2
    inpath = (r"E:/ESA/ESA_LUC_2000_SPEI03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        # print(f.variables['lccs_class'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        lcc = (f.variables['lcc'][:])
    return lcc


# %%


def tTest(list1, list2):
    t = np.zeros((30, 50))
    pt = np.zeros((30, 50))

    for r in range(30):
        if r % 30 == 0:
            print(f"column {r} is done!")
        for c in range(50):
            a = list1[:, r, c]
            b = list2[:, r, c]
            if np.isnan(a).any() or np.isnan(b).any():
                t[r, c] = np.nan
                pt[r, c] = np.nan
            else:
                levene = stats.levene(a, b, center='median')
                if levene[1] < 0.05:
                    t[r, c] = np.nan
                    pt[r, c] = np.nan
                else:
                    tTest = stats.stats.ttest_ind(a, b, equal_var=True)
                    t[r, c] = tTest[0]
                    pt[r, c] = tTest[1]

    return t, pt

# %%


def make_map(ax):
    # 打开边框
    ax.spines['geo'].set_visible(True)

    # 设置cfeature自带底图
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1)
    ax.add_feature(cfeature.RIVERS.with_scale('50m'), linewidth=1, zorder=2)

    # 设置shp文件
    shp_path1 = r'E:\SHP\gadm36_CHN_shp\gadm36_CHN_1.shp'
    shp_path2 = r'E:/SHP/world_shp/world_adm0_Project.shp'

    provinces = cfeature.ShapelyFeature(
        Reader(shp_path1).geometries(),
        ccrs.PlateCarree(),
        edgecolor='k',
        facecolor='none')

    world = cfeature.ShapelyFeature(
        Reader(shp_path2).geometries(),
        ccrs.PlateCarree(),
        edgecolor='k',
        facecolor='none')
    ax.add_feature(provinces, linewidth=1.2, zorder=3)
    ax.add_feature(world, linewidth=1.2, zorder=3)

    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    yticks = np.arange(42, 55, 5)
    xticks = np.arange(102, 130, 5)

    ax.set_yticks(yticks)
    ax.set_xticks(xticks)  # 需要显示的纬度
    ax.tick_params(labelsize=15)

    # 区域
    region = [100.25, 124.25, 40.25, 54.75]
    ax.set_extent(region, crs=ccrs.PlateCarree())

    # 设置网格点
    ylocs = list(yticks)
    xlocs = list(xticks)
    lb = ax.gridlines(draw_labels=True, ylocs=ylocs, xlocs=xlocs, x_inline=False, y_inline=False,
                      linewidth=1, color='lightgray', linestyle='--', alpha=0.8)  # alpha是透明度

    lb.top_labels = False
    lb.bottom_labels = False
    lb.right_labels = False
    lb.left_labels = False
    lb.xlabel_style = {'size': 15}  # 修改经纬度字体大小
    lb.ylabel_style = {'size': 15}

    return ax


def Creat_figure(data1, data2, data3, data4, levels, cmap, lat, lon, title):
    fig = plt.figure(figsize=(14, 4), dpi=1080)
    proj = ccrs.PlateCarree()

    fig.subplots_adjust(left=0.12, bottom=0.12, right=0.88,
                        top=0.9, wspace=0.1, hspace=0.15)

    lon, lat = np.meshgrid(lon, lat)

    title_str = ["(a) 1982-1999 Ave", "(b) 2000-2018 Ave", "(c) DIFF(b-a)"]
    for i in range(1, 4):
        ax = fig.add_subplot(1, 3, i, projection=proj)
        ax = make_map(ax)
        ax.set_title(f"{title_str[i-1]}", fontsize=15, loc="left")
        lcc2 = np.ma.array(lcc, mask=lcc != 200)
        ax.contourf(lon, lat, lcc2,
                    transform=ccrs.PlateCarree(),
                    colors="lightgray", zorder=2, corner_mask=False)
        if i == 1:
            cs = ax.contourf(lon, lat, data1,
                             levels=levels, transform=ccrs.PlateCarree(),
                             cmap=cmap, extend="both", zorder=1, corner_mask=False)
        elif i == 2:
            ax.set_yticklabels([])
            cs = ax.contourf(lon, lat, data2,
                             levels=levels, transform=ccrs.PlateCarree(),
                             cmap=cmap, extend="both", zorder=1, corner_mask=False)
        else:
            ax.set_yticklabels([])
            levels2 = np.arange(-0.4, 0.41, 0.05)
            cmap2 = cmaps.MPL_BrBG[20:109]
            cs2 = ax.contourf(lon, lat, data3,
                              levels=levels2, transform=ccrs.PlateCarree(),
                              cmap=cmap2, extend="both", zorder=1, corner_mask=False)
            ax.contourf(lon, lat, data4,
                        levels=[0, 0.05, 1], transform=ccrs.PlateCarree(),
                        hatches=['..', None], colors="none", zorder=2)

    ######
    cbar_ax = fig.add_axes([0.2, 0.08, 0.3, 0.04])
    cb = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
    cb.ax.tick_params(labelsize=15)
    ######
    cbar_ax = fig.add_axes([0.66, 0.08, 0.2, 0.04])
    cb = fig.colorbar(cs2, cax=cbar_ax, orientation='horizontal')
    cb.ax.tick_params(labelsize=15)

    # plt.suptitle(f'{title}', fontsize=20)
    plt.savefig(
        rf'E:/LAI4g/JPG_MG/Rmax(Scale)/{title}.jpg', bbox_inches='tight')
    plt.show()


# %% 画图


def read_plot(data1, data2, data3, data4, title):
    print('5 percentile is: ', np.nanpercentile(data1, 5))
    print('95 percentile is: ', np.nanpercentile(data1, 95), "\n")
    
    print("DIFF")
    print('5 percentile is: ', np.nanpercentile(data3, 5))
    print('95 percentile is: ', np.nanpercentile(data3, 95), "\n")

    levels = np.arange(-0.2, 0.61, 0.05)
    cmap = cmaps.MPL_YlGn[:113]
    Creat_figure(data1, data2, data3, data4, levels, cmap, lat, lon, title)


# %%
lcc = read_lcc()
win = 30
inpath = rf"E:/LAI4g/data_MG/Rmax(Scale)/LAI_SPEI_MG_MonthWindows{win}.nc"
r, _, _ = read_nc(inpath)

# mask部分
r_1 = r[:90, :, :]
r_2 = r[90:, :, :]

r_1_plot, r_2_plot = np.nanmean(r_1, axis=0), np.nanmean(r_2, axis=0)

r_diff = r_2_plot-r_1_plot
_, pt = tTest(r_1, r_2)

read_plot(r_1_plot, r_2_plot, r_diff, pt, title=fr"LAI Rmax Comparison")
