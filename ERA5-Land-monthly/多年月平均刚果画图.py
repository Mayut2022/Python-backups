# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 11:05:15 2022

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
    global lat, lon, et
    inpath = rf"E:/ERA5-Land-monthly/africa Total evporation 1982-2020/congo_ET_month_ave_82-20.nc"
    with nc.Dataset(inpath, mode='r') as f:
        
        #print(f.variables.keys())
        
        et = (f.variables['E'][:])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        
    et[et==32766990]=np.nan
    month = df['平年1']
    for i, mn in enumerate(month):
        et[i, :, :] = et[i, :, :]*mn
        
def read_nc_ex():
    global et_ex, t
    inpath = rf"E:/ERA5-Land-monthly/africa Total evporation 1982-2020/congo Total evporation 1982-2020.nc"
    with nc.Dataset(inpath, mode='r') as f:
        #print(f.variables.keys())
        
        time = (f.variables['time'][264:276])
        t = nc.num2date(time, 'hours since 1900-01-01 00:00:0.0').data
        
        et_ex = (f.variables['ET'][264:276, :, :])
        et_ex = et_ex*1000*(-1)
    
    month = df['平年1']
    for i, mn in enumerate(month):
        et_ex[i, :, :] = et_ex[i, :, :]*mn
    
        
#%%
def CreatMap(data, lon, lat, levels, cmap, title):
    fig = plt.figure(figsize=(10, 6), dpi=1080)
    proj = ccrs.PlateCarree()

    ax = fig.add_subplot(111, projection=proj)
    fig.subplots_adjust(left=0.2, bottom=0.2, right=0.8,
                        top=0.85, wspace=None, hspace=None)

    #ax = fig.add_axes([0.1, 0.8, 0.5, 0.3], projection=proj)

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

    ax.set_xticks(list(range(15, 31, 5)))
    ax.set_yticks(list(range(-6, 5, 2)))  # 需要显示的纬度
    ax.tick_params(labelsize=15)

    # 区域
    region = [14.125, 30.875, -5.875, 4.875]
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

    cbar_ax = fig.add_axes([0.3, 0.08, 0.4, 0.03])
    cb = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
    cb.set_label('ET (units: mm/month)', fontsize=15)
    cb.ax.tick_params(labelsize=None)
    
    
    ax.text(28.5, -10, "ERA5-LAND", color='dimgray')
    
    plt.suptitle(f"{title}", fontsize=20)
    plt.savefig(rf'E:/ERA5-Land-monthly/JPG_africa/{title}.jpg', bbox_inches='tight')
    plt.show()

#%%
df = pd.read_excel('E:/ERA5/每月天数.xlsx')
mn_str = df['月份']
#%%
read_nc()
read_nc_ex()
'''
def PLOT1(data):
    print('5 percentile is: ' , np.nanpercentile(data, 5))
    print('95 percentile is: ' , np.nanpercentile(data, 95))
    print(mn)
    print('')
    levels = np.arange(60, 121, 5)
    cmap = "Blues"
    
    CreatMap(data, lon, lat, levels,
           cmap, title=f"1982-2020 {mn_str[mn]} ET ave")
    
for mn in range(12):
    PLOT1(et[mn, :, :])
'''

#%%
def my_cmap(cmap):
    cmap = plt.get_cmap(cmap)
    newcolors = cmap(np.linspace(0, 1, 96))

    red = newcolors[:48:2]
    blue = newcolors[48:72:3]
    new = np.vstack((red, blue))
    
    newcmap = ListedColormap(new)
    plt.cm.register_cmap(name='mycmp', cmap=newcmap)
    
    return newcmap

def PLOT2(data):
    print('5 percentile is: ' , np.nanpercentile(data, 5))
    print('95 percentile is: ' , np.nanpercentile(data, 95))
    print(mn)
    print('')
    levels = np.arange(-24, 8.1, 2)
    cmap = my_cmap("RdBu")
    
    CreatMap(data, lon, lat, levels,
           cmap, title=f"2004 {mn_str[mn]} ET anomaly")
    
for mn in range(12):
    et_anom = et_ex[mn, :, :]-et[mn, :, :]
    PLOT2(et_anom)

