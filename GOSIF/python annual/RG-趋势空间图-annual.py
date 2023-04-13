# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 19:43:19 2022

@author: MaYutong
"""
# %%
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

import datetime as dt
import netCDF4 as nc
from matplotlib import font_manager
import matplotlib.dates as mdate
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
import xarray as xr

# %%


def lcc_up(data):
    data = xr.DataArray(data, dims=['y', 'x'], coords=[lat2, lon2])
    data_up = data.interp(y=lat, x=lon, method="nearest")
    return np.array(data_up)


def read_nc():
    global lcc, lat2, lon2
    # global a1, a2, o1, o2
    inpath = (r"E:\ESA\ESA_LUC_2000_SPEI03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        # print(f.variables['lccs_class'])
        lat2 = (f.variables['lat'][:])
        lon2 = (f.variables['lon'][:])
        lcc = (f.variables['lcc'][:])
        lcc = lcc_up(lcc)


def read_nc2(inpath):
    global s, p, lat, lon

    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        s = (f.variables['s'][:])
        p = (f.variables['p'][:])
        

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

    ax.set_yticks(list(range(40, 56, 5)))
    ax.set_xticks(list(range(100, 151, 5)))  # 需要显示的纬度
    ax.tick_params(labelsize=15)

    # 区域
    # region = [105.25, 124.25, 37.25, 45.75]
    region = [100, 125, 40, 55]
    ax.set_extent(region, crs=ccrs.PlateCarree())

    # 设置网格点
    ylocs = list(range(40, 56, 5))
    xlocs = list(range(100, 151, 5))
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
                     cmap=cmap, extend="both", zorder=1)

    cs2 = ax.contourf(lon, lat, data2,
                      levels=[0, 0.05, 1], transform=ccrs.PlateCarree(),
                      hatches=['..', None], colors="none", zorder=3)
    
    cs3 = ax.contourf(lon, lat, s2, colors="lightgray",
                     transform=ccrs.PlateCarree(),
                      zorder=1)

    cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.03])
    cb = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
    # cb.set_label('Significant Level: 95%', fontsize=15)

    plt.suptitle(f'{title}', fontsize=20)
    plt.savefig(rf'E:/Gosif_annual/JPG_RG/{title}.jpg', bbox_inches='tight')
    plt.show()


# %%


def read_plot(data1, data2):

    print('5 percentile is: ', np.nanpercentile(data1, 5))
    print('95 percentile is: ', np.nanpercentile(data1, 95), "\n")

    levels = np.arange(-0.003, 0.0031, 0.0005)
    cmap = cmaps.vegetation_ClarkU[40:215]
    # cmap = cmaps.MPL_BrBG[20:109]
    # title = f"MG SIF Annual 01-20 Trend"
    title = f"MG NDVI GSL 82-15 Trend"

    CreatMap(data1, data2, lon, lat, levels,
             cmap, title)


# inpath = (rf"E:/Gosif_annual/01_20/GOSIF_01_20_RG_Trend.nc")
# read_nc2(inpath)
# read_nc()
# s2 = s.copy()
# p[lcc==200]=np.nan
# s2[lcc!=200]=np.nan
# read_plot(s, p)

# %%
inpath = (rf"E:/GIMMS_NDVI/data_RG/NDVI_82_15_RG_GSL_Trend.nc")
read_nc2(inpath)
read_nc()
s2 = s.copy()
p[lcc==200]=np.nan
s2[lcc!=200]=np.nan
read_plot(s, p)
