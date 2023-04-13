# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 16:25:28 2022

@author: MaYutong
"""
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
import numpy as np
import pandas as pd
import xarray as xr

#%%
def read_nc():
    global lcc, lat, lon, df
    # global a1, a2, o1, o2
    inpath = (r"E:\ESA\ESA_LUC_2000_SPEI03.nc")
    with nc.Dataset(inpath) as f:
        print(f.variables.keys())
        # print(f.variables['lccs_class'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        lcc = (f.variables['lcc'][:])
        


# %%
def CreatMap(data, lon, lat, levels, cmap, labels, savename):
    fig = plt.figure(figsize=(14, 6), dpi=1080)
    proj = ccrs.PlateCarree()
    ax = fig.add_subplot(111, projection=proj)
    fig.subplots_adjust(left=0.1, bottom=0.2, right=0.9,
                        top=0.9, wspace=None, hspace=None)

    # 打开边框
    ax.spines['geo'].set_visible(True)

    # 设置cfeature自带底图
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1)

    # 设置shp文件
    shp_path1 = r'E:\SHP\gadm36_CHN_shp\gadm36_CHN_1.shp'
    shp_path2 = r'E:/SHP/world_shp/world_adm0_Project.shp'

    provinces = cfeature.ShapelyFeature(
        Reader(shp_path1).geometries(),
        ccrs.PlateCarree(),
        edgecolor='gray',
        facecolor='none')

    world = cfeature.ShapelyFeature(
        Reader(shp_path2).geometries(),
        ccrs.PlateCarree(),
        edgecolor='gray',
        facecolor='none')
    ax.add_feature(provinces, linewidth=0.8, zorder=3)
    ax.add_feature(world, linewidth=0.8, zorder=3)

    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    ax.set_yticks(list(range(40, 56, 5)))
    ax.set_xticks(list(range(100, 136, 5)))  # 需要显示的纬度
    ax.tick_params(labelsize=15)

    # 区域
    #region = [70, 140, 15, 55]
    #region = [-180, 180, -90, 90]
    #region = [-18, 52, -35, 37]
    #region = [100, 150, 30, 75]
    region = [100, 135, 40, 55]
    #region = [100, 135, 55, 75]
    ax.set_extent(region, crs=ccrs.PlateCarree())

    # 设置网格点
    ylocs = [50, 45]
    xlocs = np.arange(105, 131, 5)
    lb = ax.gridlines(draw_labels=True, ylocs=ylocs, xlocs=xlocs, x_inline=False, y_inline=False,
                      linewidth=1, color='k', linestyle='--', alpha=0.8)  # alpha是透明度

    lb.top_labels = False
    lb.bottom_labels = False
    lb.right_labels = False
    lb.left_labels = False
    lb.xlabel_style = {'size': 15}  # 修改经纬度字体大小
    lb.ylabel_style = {'size': 15}

    # 绘制填色图等
    lon, lat = np.meshgrid(lon, lat)
    cs = ax.contourf(lon, lat, data,
                     levels=levels, transform=ccrs.PlateCarree(),
                     colors=cmap, extend=None, zorder=1, origin="lower")


    # 绘制矩形图例
    rectangles = [Rectangle((0, 0,), 1, 1, facecolor=x) for x in colors]
    labels = labels
    ax.legend(rectangles, labels,
              bbox_to_anchor=(1.2, 0.35), fancybox=True, frameon=True)
    
    row = [52.5, 47.5, 42.5]
    column = np.arange(102.5, 135, 5)
    kw = dict(horizontalalignment="center",
              verticalalignment="center", 
              fontsize="25")
    i = 1
    for r in row:
        for c in column:
            ax.text(c, r, f"{i}", **kw)
            i += 1


    plt.suptitle(str(savename), fontsize=20)
    plt.savefig(r'E:/ESA/JPG/%s.jpg' % savename)
    plt.show()

#%%
read_nc()
df = pd.read_excel("Region1 ESA LUC降尺度SPEI03.xlsx", sheet_name='LUC Class')
colors = df['colors']

lev1 = [0, 35, 95, 140, 175, 205, 220]
cmap = colors
labels = df["classess"]
CreatMap(lcc, lon, lat, lev1, cmap, labels,
         savename="NorthEast Asia region1 LUC 2000 sub-region")