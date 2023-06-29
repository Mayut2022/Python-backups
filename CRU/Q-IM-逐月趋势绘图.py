# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 17:47:48 2022

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
from scipy.stats import linregress
plt.rcParams['font.family'] = 'Times New Roman'

#%%
def read_nc1():
    global lat, lon
    inpath = r"E:/CRU/Q_DATA_CRU-GLEAM/Q_global_81_20.nc"
    with nc.Dataset(inpath) as f:
        '''
        print(f.variables.keys())
        print(f.variables['SoilMoi0_10cm_inst']) # units: 1 kg m-2 = 1 mm
        print("")
        '''
        
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        q = (f.variables['Q'][:])
        
        return q
#%%
def IM_xarray(band1):
    t = np.arange(480)
    q=xr.DataArray(band1,dims=['t','y','x'],coords=[t, lat, lon])
    q_rg = q.loc[:, 35:60, 100:150]
    
    lat_rg = q_rg.y
    lon_rg = q_rg.x
    q_rg = np.array(q_rg)
    return q_rg, lon_rg, lat_rg

#%% 480(输入data) -> 月 年
def mn_yr(data):
    sm_mn = []
    for mn in range(12):
        sm_ = []
        for yr in range(40):
            sm_.append(data[mn])
            mn += 12
        sm_mn.append(sm_)
            
    sm_mn = np.array(sm_mn)
    
    return sm_mn

#%%
def trend(data):
    t = np.arange(1, 41, 1)
    s, r0, p = np.zeros((12, 50, 100)), np.zeros((12, 50, 100)), np.zeros((12, 50, 100))
    for mn in range(12):
        ind = 0
        for r in range(50):
            if r%30 == 0:
                print(f"{r} is done!")
            for c in range(100):
                a = data[mn, :, r, c]
                if np.isnan(a).any():
                    s[mn, r, c], r0[mn, r, c], p[mn, r, c] = np.nan, np.nan, np.nan
                    ind += 1
                else:
                    s[mn, r, c], _, r0[mn, r, c], p[mn, r, c], _  = linregress(t, a)
        print(f"month {mn} is done! \n", ind)
    return s, p
#%%
q = read_nc1()
q_IM, lon_IM, lat_IM = IM_xarray(q)
q_mn = mn_yr(q_IM)
s, p = trend(q_mn)

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
    
    '''
    ax.set_yticks(list(range(38, 48, 2)))
    ax.set_xticks(list(range(100, 151, 5)))  # 需要显示的纬度
    '''
    ax.set_yticks(list(range(35, 61, 5)))
    ax.set_xticks(list(range(100, 151, 10)))  # 需要显示的纬度
    
    ax.tick_params(labelsize=15)

    # 区域
    
    #region = [105, 125, 37, 46]
    region = [100, 150, 35, 60]
    ax.set_extent(region, crs=ccrs.PlateCarree())

    # 设置网格点
    ylocs = list(range(35, 61, 5))
    xlocs = list(range(100, 151, 10))
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

    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9,
                        top=0.95, wspace=None, hspace=0.05)
    
    lon, lat = np.meshgrid(lon, lat)
    df = pd.read_excel('E:/ERA5/每月天数.xlsx')
    mn_str = df['月份']
    for mn in range(1, 13):
        ax = fig.add_subplot(4, 3, mn, projection=proj)
        ax = make_map(ax)
        ax.set_title(f"{mn_str[mn-1]}", fontsize=15)
        cs = ax.contourf(lon, lat, data1[mn-1, :, :],
                         levels=levels, transform=ccrs.PlateCarree(),
                         cmap=cmap, extend="both", zorder=1)
        cs2 = ax.contourf(lon, lat, data2[mn-1, :, :],
                         levels=[0, 0.05, 1], transform=ccrs.PlateCarree(),
                         hatches=['.', None],colors="none", zorder=3)
        
    cbar_ax = fig.add_axes([0.3, 0.06, 0.4, 0.02])
    cb = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
    cb.set_label('Units: mm/month (CRU-GLEAM)', fontsize=15)
    cb.ax.tick_params(labelsize=15)
    
    plt.suptitle(f'{title}', fontsize=25)
    plt.savefig(rf'E:/CRU/JPG_Q/{title}.jpg', bbox_inches = 'tight')
    plt.show()

#%%
def q_xarray(band1):
    t = np.arange(12)
    q=xr.DataArray(band1,dims=['t','y','x'],coords=[t, lat_IM, lon_IM])
    return q.loc[:, 37:46, 105:125]

#%% 画图
def read_plot2(data1, data2):
    data = q_xarray(data1)
    print('5 percentile is: ' , np.nanpercentile(data, 5))
    print('95 percentile is: ' , np.nanpercentile(data, 95), "\n")
    
    levels = np.arange(-0.6, 0.61, 0.05)
    #levels = np.arange(-2.1, 2.1, 0.025)
    cmap = cmaps.BlueRed_r
    Creat_figure(data1, data2, levels, cmap, lat_IM, lon_IM, title=f"WaterYield Monthly Trend 81-20")

    
read_plot2(s, p)
