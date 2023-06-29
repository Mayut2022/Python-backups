# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 20:47:29 2022

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

#%%
def read_nc(inpath):
    global lat, lon, s, p
    with nc.Dataset(inpath, mode='r') as f:
        
        s = (f.variables['s'][:])
        p = (f.variables['p'][:])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        
#%%
def CreatMap(data1, data2, lon, lat, levels, cmap, title):
    fig = plt.figure(figsize=(16, 6), dpi=1080)
    proj = ccrs.PlateCarree()

    ax = fig.add_subplot(111, projection=proj)
    fig.subplots_adjust(left=0.2, bottom=0.2, right=0.8,
                        top=0.9, wspace=None, hspace=None)

    #ax = fig.add_axes([0.1, 0.8, 0.5, 0.3], projection=proj)

    # 打开边框
    ax.spines['geo'].set_visible(True)

    # 设置cfeature自带底图
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1)
    #ax.add_feature(cfeature.RIVERS.with_scale('50m'), linewidth=1.5, zorder=3)

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
    ax.add_feature(provinces, linewidth=0.8, zorder=3)
    ax.add_feature(world, linewidth=0.8, zorder=2)

    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    
    '''
    ax.set_yticks(list(range(42, 55, 3)))
    ax.set_xticks(list(range(100, 136, 5)))  # 需要显示的纬度
    
    ax.set_yticks(list(range(-80, 81, 20)))
    ax.set_xticks(list(range(-180, 181, 45)))  # 需要显示的纬度
    '''
    ax.set_yticks(list(range(-80, 81, 20)))
    ax.set_xticks(list(range(-180, 181, 30)))
    
    ax.tick_params(labelsize=12)

    # 区域
    # region = [-180, 180, -90, 90]
    # region = [100.125, 134.875, 40.125, 54.875]
    region = [0, 180, 30, 90]
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
    cs = ax.contourf(lon, lat, data1,
                     levels=levels, transform=ccrs.PlateCarree(),
                     cmap=cmap, extend="both", zorder=1)
    
    cs2 = ax.contourf(lon, lat, data2,
                     levels=[0, 0.05, 1], transform=ccrs.PlateCarree(),
                     hatches=['....', None],colors="none", zorder=3)

    cbar_ax = fig.add_axes([0.3, 0.08, 0.4, 0.03])
    cb = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
    cb.set_label('Significant Level: 95%', fontsize=15)
    #cb.ax.tick_params(labelsize=None)
    ax.text(135, -140, "GLEAM v3.6", color='dimgray', fontsize=12)
    #ax.text(130, 35, "MERRA2", color='dimgray')
    plt.suptitle(f"{title}", fontsize=20)
    #plt.savefig(rf'E:/Gleamv3.6a/v3.6a/JPG_GLOBAL/SPEI-ET相关/{title}.jpg', 
     #           bbox_inches='tight')
    plt.savefig(rf'E:/Gleamv3.6a/v3.6a/JPG_GLOBAL/SPEI-ET相关/EUROASIA/{title}.jpg', 
                bbox_inches='tight')
    plt.show()

#%%
df = pd.read_excel('E:/ERA5/每月天数.xlsx')
mn_str = df['月份']

#%%
def PLOT1(data1, data2):
    
    print('5 percentile is: ' , np.nanpercentile(data1, 5))
    print('95 percentile is: ' , np.nanpercentile(data1, 95))
    
    
    levels = np.arange(-0.025, 0.025, 0.0025)
    cmap = "RdYlBu"
    
    CreatMap(data1, data2, lon, lat, levels,
        cmap, title=f"81-20 SPEI-T CORR Trend (n=5) GLEAM {mn_str[mn-1]}")
     
#%%
for mn in range(1, 13):
    inpath = rf'E:/Gleamv3.6a/v3.6a/global/Corr/SPEI_T_81_20_MovingTrend_(n=5)_month{mn}.nc'
    read_nc(inpath)
    PLOT1(s, p)
