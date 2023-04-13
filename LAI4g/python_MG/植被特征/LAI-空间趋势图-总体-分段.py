# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 13:43:26 2023

@author: MaYutong
"""

import cmaps
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.io.shapereader import Reader, natural_earth

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.ticker as mticker
from matplotlib.patches import Rectangle

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
        lai1, lai2 = lai_gs[:18], lai_gs[18:]
    return lai_gs, lai1, lai2


# %%


def trend(data):
    sp = data.shape[0]
    t = np.arange(sp)
    s, p = np.zeros((30, 50)), np.zeros((30, 50))

    for r in range(30):
        for c in range(50):
            a = data[:, r, c]
            if np.isnan(a).any():
                s[r, c], p[r, c] = np.nan, np.nan
            else:
                s[r, c], _, _, p[r, c], _ = linregress(t, a)

    return s, p

# %%


def CreatMap(data1, data2, lon, lat, levels, cmap, title):
    fig = plt.figure(figsize=(10, 6), dpi=1080)
    proj = ccrs.PlateCarree()
    ax = fig.add_subplot(111, projection=proj)
    fig.subplots_adjust(left=0.1, bottom=0.2, right=0.9,
                        top=0.9, wspace=None, hspace=None)

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
    
    yticks = np.arange(42.5, 55, 5)
    xticks = np.arange(102.5, 130, 5)

    ax.set_yticks(yticks)
    ax.set_xticks(xticks)  # 需要显示的纬度
    ax.tick_params(labelsize=15)

    # 区域
    region = [100.25, 124.75, 40.25, 54.75]
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

    # 绘制填色图等
    lon, lat = np.meshgrid(lon, lat)

    cs = ax.contourf(lon, lat, data1,
                     levels=levels, transform=ccrs.PlateCarree(),
                     cmap=cmap, extend="both", zorder=1, corner_mask=False)

    cs2 = ax.contourf(lon, lat, data2,
                      levels=[0, 0.05, 1], transform=ccrs.PlateCarree(),
                      hatches=['..', None], colors="none", zorder=3)

    lcc2 = np.ma.array(lcc, mask=lcc != 200)
    ax.contourf(lon, lat, lcc2,
                transform=ccrs.PlateCarree(),
                colors="lightgray", zorder=2, corner_mask=False)

    cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.03])
    cb = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
    cb.set_label('Significant Level: 95%', fontsize=15)

    plt.suptitle(f'{title}', fontsize=20)
    plt.savefig(rf'E:/LAI4g/JPG_MG/{title}.jpg', bbox_inches='tight')
    plt.show()

# %%
def PLOT(data1, data2, title):
    levels = np.arange(-0.035, 0.036, 0.005)
    cmap = cmaps.MPL_BrBG[20:109]

    print(title)

    print('5 percentile is: ', np.nanpercentile(data1, 5))
    print('95 percentile is: ', np.nanpercentile(data1, 95), "\n")


    CreatMap(data1, data2, lon, lat, levels, cmap, title)

# %%
lcc = read_lcc()
lai, lai1, lai2 = read_nc()
s0, p0 = trend(lai)
s1, p1 = trend(lai1)
s2, p2 = trend(lai2)

title = ["LAI Trend 1982-2020", "LAI Trend 1982-1999", "LAI Trend 2000-2020"]

for i, ti in enumerate(title[:]):
    exec(f"PLOT(s{i}, p{i}, ti)")
