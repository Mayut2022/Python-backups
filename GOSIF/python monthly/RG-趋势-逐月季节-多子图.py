# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 20:46:06 2022

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
plt.rcParams['font.family'] = 'Times New Roman'
#%%
def read_nc():
    global sif, lat, lon
    inpath = r"E:/Gosif_Monthly/data_RG/GOSIF_01_20_RG.nc"
    with nc.Dataset(inpath) as f:
        #print(f.variables.keys())
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        sif = (f.variables['sif'][:])
        
        sif = sif*0.0001
        sif[sif>3] = np.nan
        
def read_nc2():
    global s, p
    inpath = r"E:/Gosif_Monthly/data_RG/GOSIF_01_20_RG_Trend.nc"
    with nc.Dataset(inpath) as f:
        #print(f.variables.keys())
        s = (f.variables['s'][:])
        p = (f.variables['p'][:])
        
#%%
def sif_xarray(data):
    t = np.arange(1, 13, 1)
    sif=xr.DataArray(data, dims=['t','y','x'], coords=[t, lat, lon])
    sif_rg = sif.loc[:, 55:40, 100:125]
    sif_rg = np.array(sif_rg)
    return sif_rg.data

#%%
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
    
    
    ax.set_yticks(list(range(40, 56, 5)))
    ax.set_xticks(list(range(100, 151, 5)))  # 需要显示的纬度
    ax.tick_params(labelsize=15)
    '''
    ax.set_yticks(list(range(35, 61, 5)))
    ax.set_xticks(list(range(100, 151, 10)))  # 需要显示的纬度
    '''
    ax.tick_params(labelsize=15)

    # 区域
    #region = [105.25, 124.25, 37.25, 45.75]
    region = [100, 125, 40, 55]
    #region = [100, 150, 35, 60]
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
    
    return ax



def Creat_figure(data1, data2, levels, cmap, lat, lon, title):
    fig=plt.figure(figsize=(18, 12))
    proj = ccrs.PlateCarree()

    fig.subplots_adjust(left=0.12, bottom=0.12, right=0.88,
                        top=0.9, wspace=0.05, hspace=0.27)
    
    lon, lat = np.meshgrid(lon, lat)
    df = pd.read_excel('E:/ERA5/每月天数.xlsx')
    mn_str = df['月份']
    for mn in range(1, 13):
        ax = fig.add_subplot(4, 3, mn, projection=proj)
        ax = make_map(ax)
        ax.set_title(f"{mn_str[mn-1]}", fontsize=15, loc="left")
        cs = ax.contourf(lon, lat, data1[mn-1, :, :],
                         levels=levels, transform=ccrs.PlateCarree(),
                         cmap=cmap, extend="both", zorder=1)
        cs2 = ax.contourf(lon, lat, data2[mn-1, :, :],
                         levels=[0, 0.05, 1], transform=ccrs.PlateCarree(),
                         hatches=['..', None],colors="none", zorder=3)
        
    cbar_ax = fig.add_axes([0.3, 0.06, 0.4, 0.02])
    cb = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
    cb.set_label('Units: W m−2 μm−1 sr−1', fontsize=15)
    cb.ax.tick_params(labelsize=15)
    
    plt.suptitle(f'{title}', fontsize=25)
    plt.savefig(rf'E:/Gosif_Monthly/JPG_RG/{title}.jpg', bbox_inches = 'tight')
    plt.show()
        
        
#%%
read_nc()
sif_mn = np.nanmean(sif, axis=0)
def read_plot(data1):
    global data
    data = sif_xarray(data1)
    
    print('5 percentile is: ' , np.nanpercentile(data, 5))
    print('95 percentile is: ' , np.nanpercentile(data, 95), "\n")
    
    levels = np.arange(-0.05, 0.41, 0.025)
    
    cmap = cmaps.NOC_ndvi
    data2 = np.ones((12, 500, 1001))
    Creat_figure(data1, data2, levels, cmap, lat, lon, title=f"MG SIF Monthly Mean 01-20")
    
read_plot(sif_mn)
#%%
read_nc2()

def read_plot2(data1, data2):
    data = sif_xarray(data1)
    print('5 percentile is: ' , np.nanpercentile(data, 5))
    print('95 percentile is: ' , np.nanpercentile(data, 95), "\n")
    
    levels = np.arange(-0.004, 0.0041, 0.0005)
    cmap = cmaps.vegetation_ClarkU[40:215]
    Creat_figure(data1, data2, levels, cmap, lat, lon, title=f"MG SIF Monthly Trend 01-20")
    
read_plot2(s, p)