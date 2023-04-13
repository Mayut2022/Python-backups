# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 21:21:09 2022

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

#%%
def read_nc():
    global lat, lon, e
    inpath = rf"E:/ERA5-Land-monthly/africa Total evporation 1982-2020/africa_ET_month_ave_82-20.nc"
    with nc.Dataset(inpath, mode='r') as f:
        
        print(f.variables.keys())
        
        e = (f.variables['E'][:])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        
# %%
def CreatMap(data, lon, lat, levels, cmap, title):
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
        edgecolor='k',
        facecolor='none')

    world = cfeature.ShapelyFeature(
        Reader(shp_path2).geometries(),
        ccrs.PlateCarree(),
        edgecolor='k',
        facecolor='none')
    #ax.add_feature(provinces, linewidth=0.8, zorder=3)
    ax.add_feature(world, linewidth=0.8, zorder=2)

    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    ax.set_xticks(list(range(-10, 51, 20)))
    ax.set_yticks(list(range(-30, 31, 20)))  # 需要显示的纬度
    ax.tick_params(labelsize=15)

    # 区域
    region = [-18, 52, -35, 37]
    ax.set_extent(region, crs=ccrs.PlateCarree())

    # 设置网格点
    lb = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False,
                      linewidth=0.5, color='lightgray', linestyle='--', alpha=0.8)  # alpha是透明度

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
                     cmap=cmap, extend="both", zorder=1)

    cbar_ax = fig.add_axes([0.35, 0.08, 0.3, 0.025])
    cb = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
    #cb.ax.set_xticklabels([f'{x-0.25:.2f}' for x in levels])
    cb.ax.tick_params(labelsize=None)
    cb.set_label('ET (units: mm/month)', fontsize=15)
    
    # select 矩形关键区区域
    RE = Rectangle((14, -6), 17, 11, linewidth=1, linestyle='-', zorder=3,
                   edgecolor='yellow', facecolor='none', transform=ccrs.PlateCarree())
    ax.add_patch(RE)
    
    ax.text(28.5, -10, "ERA5-LAND", color='dimgray')
    
    # 画子区域刚果雨林，这一步是新建一个ax，设置投影
    sub_ax = fig.add_axes([0.6, 0.11, 0.14, 0.155],
                          projection=ccrs.PlateCarree())
    # 画海，陆地，河流，湖泊
    sub_ax.add_feature(cfeature.OCEAN.with_scale('50m'))
    sub_ax.add_feature(cfeature.LAND.with_scale('50m'))
    sub_ax.add_feature(cfeature.RIVERS.with_scale('50m'))
    sub_ax.add_feature(cfeature.LAKES.with_scale('50m'))

    # 框区域
    sub_ax.set_extent([14, 31, -6, 5])
    sub_ax.add_feature(world, linewidth=0.8, zorder=2)
    plt.title('Congo forest')
    cs2 = sub_ax.contourf(lon, lat, data,
                          levels=levels, transform=ccrs.PlateCarree(),
                          cmap=cmap, extend="both", zorder=1)

    plt.suptitle(f"{title}", fontsize=20)
    plt.savefig(rf'E:/ERA5-Land-monthly/JPG_africa/{title}.jpg', bbox_inches='tight')
    plt.show()
        
#%%
df = pd.read_excel('E:/ERA5/每月天数.xlsx')
mn_str = df['月份']

#%%
read_nc()
e[e==32766990]=np.nan
month = df['平年1']
for i, mn in enumerate(month):
    e[i, :, :] = e[i, :, :]*mn
    
#%%

def PLOT1(data):
    print('5 percentile is: ' , np.nanpercentile(data, 5))
    print('95 percentile is: ' , np.nanpercentile(data, 95))
    print(mn)
    print('')
    levels = np.arange(0, 121, 10)
    cmap = "Blues"
    
    CreatMap(data, lon, lat, levels,
             cmap, title=f"1982-2020 {mn_str[mn]} ET ave")
    
for mn in range(12):
    PLOT1(e[mn, :, :])
