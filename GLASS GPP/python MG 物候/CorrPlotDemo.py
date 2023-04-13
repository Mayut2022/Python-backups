# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 20:48:31 2022

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
from matplotlib import cm
from matplotlib import font_manager
import matplotlib.dates as mdate
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
from scipy.stats import pearsonr
import xarray as xr

plt.rcParams['font.sans-serif'] = ['times new roman'] # 指定默认字体

#%%
def read_nc():
    global spei, lat, lon
    inpath = (rf"/mnt/e/SPEI_base/data/SPEI03_MG.nc")
    with nc.Dataset(inpath) as f:
        #print(f.variables.keys())
        
        spei = (f.variables['spei'][960:])
        time = (f.variables['time'][960:])
        t = nc.num2date(time, 'days since 1900-01-01 00:00:0.0').data
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])

#%%
def read_nc2(inpath):
    global gpp, lat2, lon2
    with nc.Dataset(inpath) as f:
        #print(f.variables.keys())
        
        gpp = (f.variables['GPP'][:])
        lat2 = (f.variables['lat'][:])
        lon2 = (f.variables['lon'][:])
        
    return gpp

#%%
def read_nc4(inpath):
    global wue
    with nc.Dataset(inpath) as f:
        #print(f.variables.keys())
        
        wue = (f.variables['WUE'][:])
        lat2 = (f.variables['lat'][:])
        lon2 = (f.variables['lon'][:])
        
    return wue

#%%
def sif_xarray(band1):
    mn = np.arange(12)
    sif=xr.DataArray(band1, dims=['mn', 'y','x'],coords=[mn, lat2, lon2])
    
    sif_MG = sif.loc[:, 40:55, 100:125]
    return np.array(sif_MG)

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
def exact_data1():
    for yr in range(1982, 2019):
        inpath =  rf"/mnt/e/GLASS-GPP/Month RG/GLASS_GPP_RG_SPEI_0.5X0.5_{yr}.nc"
        data = read_nc2(inpath)
        
        data_MG = sif_xarray(data)
        data_MG = data_MG.reshape(1, 12, 30, 50)
        
        if yr == 1982:
            data_all = data_MG
        else:
            data_all = np.vstack((data_all, data_MG))
        
    return data_all

def exact_data3():
    for yr in range(1982, 2019):
        inpath3 =  rf"/mnt/e/GLASS-GPP/Month RG WUE/GLASS_WUE_RG_SPEI_0.5X0.5_{yr}.nc"
        data = read_nc4(inpath3)
        
        data_MG = sif_xarray(data)
        data_MG = data_MG.reshape(1, 12, 30, 50)
        
        if yr == 1982:
            data_all = data_MG
        else:
            data_all = np.vstack((data_all, data_MG))
        
    return data_all

#%%
def corr(data1, data2):
    r = np.zeros((30, 50))
    p = np.zeros((30, 50))
    for i in range(30):
        for j in range(50):
            if np.isnan(data1[:, i, j]).any() or np.isnan(data2[:, i, j]).any():
                r[i, j], p[i, j] = np.nan, np.nan
            else:
                r[i, j], p[i, j] = pearsonr(data1[:, i, j], data2[:, i, j])
                
    return r, p

#%%
def CreatMap(data1, data2, lon, lat, levels, cmap, title):
    fig = plt.figure(figsize=(10, 6), dpi=1080)
    proj = ccrs.PlateCarree()

    ax = fig.add_subplot(111, projection=proj)
    fig.subplots_adjust(left=0.1, bottom=0.2, right=0.9,
                        top=0.85, wspace=None, hspace=None)

    #ax = fig.add_axes([0.1, 0.8, 0.5, 0.3], projection=proj)

    # 打开边框
    ax.spines['geo'].set_visible(True)

    # 设置cfeature自带底图
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1)
    ax.add_feature(cfeature.RIVERS.with_scale('50m'), linewidth=1.5, zorder=3)

    # 设置shp文件
    shp_path1 = r'/mnt/e/SHP/gadm36_CHN_shp/gadm36_CHN_1.shp'
    shp_path2 = r'/mnt/e/SHP/world_shp/world_adm0_Project.shp'

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
    
    yticks = np.arange(42.5, 55, 5)
    xticks = np.arange(102.5, 125, 5)
    ax.set_yticks(yticks)
    ax.set_xticks(xticks)  # 需要显示的纬度
    '''
    ax.set_yticks(list(range(-80, 81, 20)))
    ax.set_xticks(list(range(-180, 181, 45)))  # 需要显示的纬度
    
    ax.set_yticks(list(range(-80, 81, 20)))
    ax.set_xticks(list(range(-180, 181, 30)))
    '''
    ax.tick_params(labelsize=12)

    # 区域
    #region = [-180, 180, -90, 90]
    region = [100.25, 124.75, 40.25, 54.75]
    #region = [0, 180, 30, 90]
    # region = [100, 125, 40, 55]
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

    # 绘制填色图等
    lon, lat = np.meshgrid(lon, lat)
    cs = ax.contourf(lon, lat, data1,
                     levels=levels, transform=ccrs.PlateCarree(),
                     cmap=cmap, extend="both", zorder=1)
    
    cs2 = ax.contourf(lon, lat, data2,
                         levels=[0, 0.05, 1], transform=ccrs.PlateCarree(),
                         hatches=['.', None], colors="none", zorder=3)

    cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.03])
    cb = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
    #cb.set_label('units: mm/month', fontsize=15)
    cb.ax.tick_params(labelsize=12)
    #ax.text(135, -140, "GLEAM v3.6", color='dimgray', fontsize=12)
    #ax.text(143, 30, "PRE:CRU 82-15", color='dimgray', fontsize=12)
    plt.suptitle(f"{title}", fontsize=20)
    plt.savefig(rf'/mnt/e/GLASS-GPP/JPG MG 物候/{title}.jpg', 
                bbox_inches='tight')
    plt.show()


#%% GPP
# read_nc()
# spei_mn = mn_yr(spei)
# gpp_MG = exact_data1()
# wue_MG = exact_data3()

# #%% 相关性
# # spei_gs = spei_mn[3:10, 1:38, :, :].reshape((37*7, 30, 50), order='F')
# # gpp_gs = gpp_MG[:, 3:10, :, :].reshape(37*7, 30, 50)
# # wue_gs = wue_MG[:, 3:10, :, :].reshape(37*7, 30, 50)

# # r, p = corr(spei_gs, gpp_gs)

# spei_ave = spei_mn[3:10, 1:38, :, :].mean(axis=0)
# gpp_ave = gpp_MG[:, 3:10, :, :].mean(axis=1)

# r, p = corr(spei_ave, gpp_ave)

# def plot(data1, data2):
#     print('5 percentile is: ' , np.nanpercentile(data1, 5))
#     print('95 percentile is: ' , np.nanpercentile(data1, 95), "\n")
    
#     levels = np.arange(0, 1.01, 0.05)
#     cmap = cmaps.MPL_OrRd[:97]
#     CreatMap(data1, data2, lon, lat, levels, cmap, title="SPEI-GPP CORR GS Mean")
    
    
# # plot(r, p)

# #%%
# r2, p2 = corr(spei_gs, wue_gs)
# def plot(data1, data2):
#     print('5 percentile is: ' , np.nanpercentile(data1, 5))
#     print('95 percentile is: ' , np.nanpercentile(data1, 95), "\n")
    
#     levels = np.arange(-0.3, 0.31, 0.05)
#     cmap = cmaps.BlueRed
#     CreatMap(data1, data2, lon, lat, levels, cmap, title="SPEI-WUE CORR Apr-Oct")
    
    
# # plot(r2, p2)






# #%%
# '''
# a = spei_gs[:, :, 10, 10]
# a_re = spei_gs[:, 10, 10]
# b = gpp_gs[:, :, 10, 20]
# b_re = gpp_gs[:, 10, 20]
# '''