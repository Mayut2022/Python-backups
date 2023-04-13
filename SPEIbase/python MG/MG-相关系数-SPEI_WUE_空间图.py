# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 20:40:19 2022

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

import netCDF4 as nc

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
import xarray as xr
import xlsxwriter

import seaborn as sns

from scipy.stats import linregress
from scipy.stats import pearsonr


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
def read_nc(inpath):
    global wue, lat2, lon2
    with nc.Dataset(inpath) as f:
        #print(f.variables.keys())
        
        wue = (f.variables['WUE'][:])
        lat2 = (f.variables['lat'][:])
        lon2 = (f.variables['lon'][:])
        
    return wue

def sif_xarray2(band1):
    mn = np.arange(12)
    sif=xr.DataArray(band1, dims=['mn', 'y','x'],coords=[mn, lat2, lon2])
    
    sif_MG = sif.loc[:, 40:55, 100:125]
    return np.array(sif_MG)


def exact_data3():
    for yr in range(1982, 2019):
        inpath3 =  rf"E:/GLASS-GPP/Month RG WUE/GLASS_WUE_RG_SPEI_0.5X0.5_{yr}.nc"
        data = read_nc(inpath3)
        
        data_MG = sif_xarray2(data)
        data_MG = data_MG.reshape(1, 12, 30, 50)
        
        if yr == 1982:
            data_all = data_MG
        else:
            data_all = np.vstack((data_all, data_MG))
        
    return data_all

#%%
def read_nc2(scale):
    inpath = (rf"E:/SPEI_base/data/spei{scale}.nc")
    with nc.Dataset(inpath) as f:
        #print(f.variables.keys())
        
        spei = (f.variables['spei'][960:])
        time = (f.variables['time'][960:])
        t2 = nc.num2date(time, 'days since 1900-01-01 00:00:0.0').data
        lat_g = (f.variables['lat'][:])
        lon_g = (f.variables['lon'][:])
        
    def sif_xarray(band1):
        mn = np.arange(12)
        yr = np.arange(40)
        sif=xr.DataArray(band1, dims=['mn', 'yr', 'y','x'],coords=[mn, yr, lat_g, lon_g])
        
        sif_MG = sif.loc[:, :, 40:55, 100:125]
        return np.array(sif_MG)
    
    spei_mn = mn_yr(spei)
    spei_mn_MG = sif_xarray(spei_mn)
    
    return spei_mn_MG

#%%
def read_nc3():
    global lcc, lat, lon
    # global a1, a2, o1, o2
    inpath = (r"E:\ESA\ESA_LUC_2000_SPEI03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        # print(f.variables['lccs_class'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        lcc = (f.variables['lcc'][:])

#%%
def corr(data1, data2, scale):
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
    fig.subplots_adjust(left=0.1, bottom=0.18, right=0.9,
                        top=0.92, wspace=None, hspace=None)

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
    
    
    ax.set_yticks(list(range(35, 60, 5)))
    ax.set_xticks(list(range(100, 151, 5)))  # 需要显示的纬度
    '''
    ax.set_yticks(list(range(-80, 81, 20)))
    ax.set_xticks(list(range(-180, 181, 45)))  # 需要显示的纬度
    
    ax.set_yticks(list(range(-80, 81, 20)))
    ax.set_xticks(list(range(-180, 181, 30)))
    '''
    ax.tick_params(labelsize=12)

    # 区域
    #region = [-180, 180, -90, 90]
    # region = [100.125, 134.875, 40.125, 54.875]
    #region = [0, 180, 30, 90]
    region = [100, 125, 40, 55]
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
    # plt.savefig(rf'E:/SPEI_base/python MG/JPG/WUE/{title}.jpg', 
    #             bbox_inches='tight')
    plt.show()


        
#%%
wue_MG = exact_data3()
read_nc3()


#%%
scale = [str(x).zfill(2) for x in range(1, 13)]
month = np.arange(4, 11, 1) ## Apr-Oct 生长季

df2 = pd.read_excel('E:/ERA5/每月天数.xlsx')
mn_str = df2['月份']

def plot(data1, data2, mn, s):
    print('5 percentile is: ' , np.nanpercentile(data1, 5))
    print('95 percentile is: ' , np.nanpercentile(data1, 95), "\n")
    
    levels = np.arange(-0.6, 0.61, 0.05)
    cmap = cmaps.MPL_PRGn
    CreatMap(data1, data2, lon, lat, levels, cmap, title=f"SPEI{s}-WUE CORR {mn_str[mn-1]}")
    
    
   
for mn in month:
    print(f"month {mn} is in programming!")
    wue_mn = wue_MG[:, mn-1, :, :]
    for s in [scale[0], scale[3]]:
        spei_s = read_nc2(s)
        spei_mn = spei_s[mn-1, 1:38, :,:]
        r, p = corr(wue_mn, spei_mn, s)
        print(f"scale spei{s} is done!")
        plot(r, p, mn, s) 
    print(f"Next is Cartopy plot-----")