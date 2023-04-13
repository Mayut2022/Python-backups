# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 09:10:01 2022

@author: MaYutong
"""

from collections import Counter

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
def read_nc(lat1, lat2, lon1, lon2):
    
    global a1, a2, o1, o2
    global df
    inpath = (r"E:/ESA/ESACCI-LC-L4-LCCS-Map-300m-P1Y-2000-v2.0.7cds.nc")
    with nc.Dataset(inpath) as f:
        print(f.variables.keys())
        print(f.variables['lccs_class'])
        
        lat_all = (f.variables['lat'][:])
        lon_all = (f.variables['lon'][:])
        
        ''' (75-30N, 0-180E) Global: (90N-90S, -180-180)
        a1 = np.where(lat_all<=75)[0][0]
        a2 = np.where(lat_all>=30)[-1][-1]
        o1 = np.where(lon_all>=0)[0][0]
        o2 = np.where(lon_all<=180)[-1][-1]
        '''
        a1 = np.where(lat_all<=lat1)[0][0]
        a2 = np.where(lat_all>=lat2)[-1][-1]
        o1 = np.where(lon_all>=lon1)[0][0]
        o2 = np.where(lon_all<=lon2)[-1][-1]
        
        lat = (f.variables['lat'][a1:a2])
        lon = (f.variables['lon'][o1:o2])
        
        lcc = (f.variables['lccs_class'][:, a1:a2, o1:o2])
        
        lcc = np.squeeze(lcc)
        color1 = ["#000000"]
        flag_colors = f.variables['lccs_class'].flag_colors.split(" ")
        flag_colors = color1+flag_colors
        flag_values = f.variables['lccs_class'].flag_values
        flag_meanings = f.variables['lccs_class'].flag_meanings.split(" ")
        
        df = pd.DataFrame(dict(values=flag_values, colors=flag_colors, meanings=flag_meanings))

        
    return lcc, lat, lon
    

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
    ax.add_feature(cfeature.RIVERS.with_scale('50m'), linewidth=1)

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

    ax.set_yticks(list(range(40, 61, 5)))
    ax.set_xticks(list(range(120, 151, 5)))  # 需要显示的纬度
    ax.tick_params(labelsize=15)

    # 区域
    #region = [70, 140, 15, 55]
    #region = [-180, 180, -90, 90]
    #region = [-18, 52, -35, 37]
    #region = [100, 150, 30, 75]
    #region = [100, 135, 40, 55]
    #region = [100, 135, 55, 75]
    region = [120, 150, 40, 60]
    ax.set_extent(region, crs=ccrs.PlateCarree())

    # 设置网格点
    ylocs = [55, 50, 45]
    xlocs = [130, 140]
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
    
    cs = ax.contourf(lon, lat, data,
                     levels=levels, transform=ccrs.PlateCarree(),
                     colors=cmap, extend=None, zorder=1)
    

    # 绘制矩形图例
    rectangles = [Rectangle((0, 0,), 1, 1, facecolor=x) for x in cmap]
    labels = labels
    ax.legend(rectangles, labels,
              bbox_to_anchor=(1.45, 1.15), fancybox=True, frameon=True)
    
    '''
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
    '''        
    
    plt.suptitle(str(savename), fontsize=20)
    #plt.savefig(r'E:/ESA/JPG/%s.jpg' % savenameb, box_inches='tight')
    plt.show()
   
#%%
def lcc_count(data):
    sp = data.shape
    data = data.reshape(1, sp[0]*sp[1])
    data = np.squeeze(data)
    
    return Counter(data)
        
#%%
# 输入参数为区域的[上， 下， 左，右]边界
lcc, lat, lon = read_nc(60, 40, 120, 150)

lcc_c = lcc_count(lcc)

a = dict(lcc_c)
b = sorted(a.keys())
a = sorted(a.items())
print(a)

b =list(map(int,b))
b = pd.Series(b) #uint8 类型，存储0-255，虽然为负数，但是数值相等。
b.name = "NEA"
df = pd.merge(df, b, left_on='values', right_on='NEA')

#%%
a_per = []
for x in a:
    a_per.append(x[-1]/(7199*10799))
df['NEA_per'] = a_per
    
#%% 细分类
'''
def level(values):
    lev1 = [5]
    for i, x in enumerate(values[:-1]):
        lev1.append((x+values[i+1])/2)
    lev1.append(220)   
    return lev1

colors = df['colors']
lev1 = level(b)


cmap = colors
labels = df["meanings"]

CreatMap(lcc, lon, lat, lev1, cmap, labels,
         savename="EuroAsia LUC 2000 300m")
'''

#%% 粗分类
df2 = pd.read_excel("Region1 ESA LUC降尺度SPEI03.xlsx", sheet_name='LUC Class')
colors2 = df2['colors']

lev2 = [0, 35, 95, 140, 175, 205, 220]
cmap = colors2
labels = df2["classess"]
#CreatMap(lcc, lon, lat, lev2, cmap, labels,
 #        savename="NEA LUC 2000")
 
#%% 调试图例位置
'''
fig = plt.figure(figsize=(14, 6), dpi=1080)
proj = ccrs.PlateCarree()
ax = fig.add_subplot(111, projection=proj)
colors = df['colors']
# 绘制矩形图例
rectangles = [Rectangle((0, 0,), 1, 1, facecolor=x) for x in colors]
labels = labels
ax.legend(rectangles, labels,
          bbox_to_anchor=(1.45, 1.15), fancybox=True, frameon=True)
plt.savefig(r'E:/ESA/JPG/test.jpg', )
'''