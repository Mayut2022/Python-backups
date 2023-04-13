# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 15:17:45 2022

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

from scipy.stats import linregress

#%%
def read_nc2():
    global sm_Gl, lat2, lon2
    inpath = rf"E:/GLDAS Noah/SM_81_20/region1_SM_month_ORINGINAL.nc"
    with nc.Dataset(inpath) as f:
        print(f.variables.keys())
        # print(f.variables['sm'])
        sm_Gl = (f.variables['sm'][:])
        lat2 = (f.variables['lat'][:])
        lon2 = (f.variables['lon'][:])
        sm_Gl[sm_Gl==sm_Gl[0, 0, 0, -1]] = np.nan
        # layer = (f.variables['sm'].layer)
        
#%%
def read_nc3():
    global sm_ERA, lat3, lon3
    inpath = rf"E:/ERA5-Land-monthly/REGION1 SM/region1_SM_month_ORINGINAL.nc"
    with nc.Dataset(inpath) as f:
        print(f.variables.keys())
        # print(f.variables['sm'])
        sm_ERA = (f.variables['sm'][:])
        lat3 = (f.variables['lat'][:])
        lon3 = (f.variables['lon'][:])
        sm_ERA[sm_ERA==sm_ERA[0, 0, 0, -1]] = np.nan
        # layer = (f.variables['sm'].layer)
        

#%%
read_nc2()
read_nc3()


#%%
def Trend(data, lat, lon):
    sp = data.shape
    a, b = sp[2], sp[3]
    
    t=np.arange(1, 481, 1)
    s,r,p = np.zeros((4, a, b)), np.zeros((4, a, b)), np.zeros((4, a, b))
    
    for l in range(4):
        for i in range(len(lat)):
            for j in range(len(lon)):
                s[l, i, j],_,r[l, i, j], p[l, i, j],_  = linregress(t, data[:,l, i, j])
                
    return s, r, p

s_ERA, r_ERA, p_ERA = Trend(sm_ERA, lat3, lon3)
s_Gl, r_Gl, p_Gl = Trend(sm_Gl, lat2, lon2)



#%%
'''
mask = p_ERA > 0.01
p_ERA = ma.masked_array(p_ERA, mask=mask)
a = p_ERA[-1, :, :]
'''
# %%
def CreatMap(data1, data2, lon, lat, levels, cmap, title):
    fig = plt.figure(figsize=(10, 4), dpi=1080)
    proj = ccrs.PlateCarree()

    ax = fig.add_subplot(111, projection=proj)
    fig.subplots_adjust(left=0.2, bottom=0.2, right=0.8,
                        top=0.85, wspace=None, hspace=None)

    #ax = fig.add_axes([0.1, 0.8, 0.5, 0.3], projection=proj)

    # 打开边框
    ax.spines['geo'].set_visible(True)

    # 设置cfeature自带底图
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1)
    ax.add_feature(cfeature.RIVERS.with_scale('50m'), linewidth=1.5, zorder=3)

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

    ax.set_yticks(list(range(42, 55, 3)))
    ax.set_xticks(list(range(100, 136, 5)))  # 需要显示的纬度
    ax.tick_params(labelsize=12)

    # 区域
    region = [100.125, 134.875, 40.125, 54.875]
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
                     [0, 0.01, 1], transform=ccrs.PlateCarree(),
                     hatches=['..', None], colors="none", zorder=2)

    cbar_ax = fig.add_axes([0.3, 0.08, 0.4, 0.03])
    cb = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
    cb.set_label('SM (units: kg/m2)', fontsize=12)
    #cb.ax.tick_params(labelsize=None)
    ax.text(130, 35, "GLDAS-NOAH", color='dimgray')
    plt.suptitle(f"{title}", fontsize=18)
    plt.savefig(rf'E:/ERA5-Land-monthly/趋势/{title}.jpg', bbox_inches='tight')
    plt.show()


#%% 480月 ERA
def PLOT1(data1, data2):
    data1[data1<-1] = np.nan
    print('5 percentile is: ' , np.nanpercentile(data1, 5))
    print('95 percentile is: ' , np.nanpercentile(data2, 95))
    print('')
    levels = np.arange(-0.005, 0.0051, 0.001)
    cmap = "RdBu"
    
    CreatMap(data1, data2, lon3, lat3, levels,
            cmap, title=f"1981-2020 Layer{i+1} SM Trend")
    
for i in range(4):
    #PLOT1(s_ERA[i, :, :], p_ERA[i, :, :])
    pass
    
#%% 480月 GLDAS
def PLOT1(data1, data2):
    #data1[data1<-1] = np.nan
    print('5 percentile is: ' , np.nanpercentile(data1, 5))
    print('95 percentile is: ' , np.nanpercentile(data2, 95))
    print('')
    l = np.nanpercentile(data1, 5)
    r = np.nanpercentile(data2, 95)
    levels = np.arange(-1, 1.1, 0.1)
    cmap = "RdBu"
    
    CreatMap(data1, data2, lon2, lat2, levels,
            cmap, title=f"1981-2020 Layer{i+1} SM Trend GLDAS")
    
for i in range(4):
    PLOT1(s_Gl[i, :, :], p_Gl[i, :, :])






