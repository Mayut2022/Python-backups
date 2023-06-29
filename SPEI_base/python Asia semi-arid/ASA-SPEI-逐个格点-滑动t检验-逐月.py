# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 19:29:08 2022

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
from matplotlib import cm
from matplotlib import font_manager
import matplotlib.dates as mdate
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd

import xarray as xr

#%%
def read_nc2():
    global spei, lat, lon
    inpath = (rf"E:/SPEI_base/data/spei03_ASA.nc")
    with nc.Dataset(inpath) as f:
        #print(f.variables.keys())
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        spei = (f.variables['spei'][960:])

#%%
def slidet(inputdata, step):
    inputdata = np.array(inputdata)
    n = inputdata.shape[0]
    n1 = step    #n1, n2为子序列长度，需调整
    n2 = step
    t = np.zeros(n)
    for i in range (step, n-step-1):
        x1 = inputdata[i-step : i]
        x2 = inputdata[i : i+step]
        x1_mean = np.nanmean(inputdata[i-step : i])   
        x2_mean = np.nanmean(inputdata[i : i+step])
        s1 = np.nanvar(inputdata[i-step : i])          
        s2 = np.nanvar(inputdata[i : i+step])
        s = np.sqrt((n1 * s1 + n2 * s2) / (n1 + n2 - 2))
        t[i] = (x2_mean - x1_mean) / (s * np.sqrt(1/n1+1/n2))
    t[:step]=np.nan  
    t[n-step+1:]=np.nan 
    
    return t    
 
#%%
def tip(data, thre):
    yr_tip = []
    a=False
    year = np.arange(1981, 2021, 1)
    for d, yr in zip(data, year):
        if np.isnan(d)==False:
            if d>thre or d<-thre:
                yr_tip.append(yr)
                print(yr)
                a=True
    return a, yr_tip   

#%% 分箱test
'''
for r in range(18):
    for c in range(40):
        test = spei[:, r, c]
        if np.isnan(test).any():
            pass
        else:
            t_move = slidet(test, N)
            a, yr_tip = tip(t_move, tt)
            title = f"Moving t-test"
            if a==True:
                print(title, "\n")


bin = np.arange(1980, 2021, 5)
cats = pd.cut(yr_tip, bin, right=True)
num = cats.value_counts()
a, *b, c = num
'''

# %%
def CreatMap(data1, lon, lat, levels, cmap, title):
    fig = plt.figure(figsize=(15, 6), dpi=1080)
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

    ax.set_yticks(list(range(25, 56, 5)))
    ax.set_xticks(list(range(30, 151, 10)))  # 需要显示的纬度
    ax.tick_params(labelsize=15)

    # 区域
    region = [30.25, 149.75, 25.25, 54.75]
    ax.set_extent(region, crs=ccrs.PlateCarree())

    # 设置网格点
    ylocs = np.arange(25, 56, 5)
    xlocs = np.arange(30, 151, 10)
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
    '''
    cs2 = ax.contourf(lon, lat, data2,
                     levels=[0, 0.05, 1], transform=ccrs.PlateCarree(),
                     hatches=['.', None],colors="none", zorder=3)
    '''
    
    cbar_ax = fig.add_axes([0.3, 0.08, 0.4, 0.03])
    cb = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
    #cb.set_label('Significant Level: 95%', fontsize=15)


    plt.suptitle(f'{title}', fontsize=20)
    plt.savefig(rf'E:/SPEI_base/python Asia semi-arid/JPG/{title}.jpg', bbox_inches = 'tight')
    plt.show()

#%% 480(输入data) -> 月 年
def mn_yr(data):
    tmp_mn = []
    for mn in range(12):
        tmp_ = []
        for yr in range(40):
            tmp_.append(data[mn])
            mn += 12
        tmp_mn.append(tmp_)
            
    tmp_mn = np.array(tmp_mn)
    
    return tmp_mn
#%%

read_nc2()
spei_mn = mn_yr(spei)

N = 10
tt = 2.8784

#%%
def yr_cut(N, tt, spei, mn):
    bin = np.arange(1980, 2021, 5)
    
    for i in range(1, 9):
        exec(f"year{i} = np.zeros((60, 240))")
    
    for r in range(60):
        for c in range(240):
            test = spei[:, r, c]
            if np.isnan(test).any():
                pass
            else:
                t_move = slidet(test, N)
                a, yr_tip = tip(t_move, tt)
                yr_tip_cut = pd.cut(yr_tip, bin, right=True)
                num = yr_tip_cut.value_counts()
                for i, x in enumerate(num):
                    exec(f"year{i+1}[r, c] = x")
                    
    del a, c, i, r, x, N, tt
    
    levels = np.arange(0, 6, 1)
    cmap = "tab10"
    for i, x in enumerate(num.index):
        title = f"Moving T-test Number {x} (n=10) month{mn}"
        exec(f'CreatMap(year{i+1}, lon, lat, levels, cmap, title=title)')
    
#%%
for mn in range(8, 13):
    yr_cut(N, tt, spei_mn[mn-1, :, :, :], mn)
    print(f"{mn} is done!")



