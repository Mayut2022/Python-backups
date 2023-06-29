# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 18:32:20 2023

@author: MaYutong
"""

import cmaps
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.io.shapereader import Reader, natural_earth

from eofs.standard import Eof
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.ticker as mticker
from matplotlib.patches import Rectangle

import netCDF4 as nc
import matplotlib as mpl
from matplotlib import cm
import numpy as np
import numpy.ma as ma
import pandas as pd

import xarray as xr

import warnings
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['times new roman']  # 指定默认字体

# %%
lat = np.linspace(40.25, 54.75, 30)
lon = np.linspace(100.25, 124.75, 50)


def read_lcc():
    inpath = (r"E:/ESA/ESA_LUC_2000_SPEI03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        # print(f.variables['lccs_class'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        lcc = (f.variables['lcc'][:])

    return lcc


# %%

def MG(data):
    t = np.arange(1, 481, 1)
    l = np.arange(1, 5, 1)
    spei_global = xr.DataArray(data, dims=['t', 'l', 'y', 'x'], coords=[
                               t, l, lat2, lon2])  # 原SPEI-base数据
    spei_rg1 = spei_global.loc[:, :, 40:55, 100:125]
    spei_rg1 = np.array(spei_rg1)
    return spei_rg1


def read_nc():
    global lat2, lon2
    inpath = rf"E:/GLDAS Noah/DATA_RG/SM_81_20_ORINGINAL_SPEI0.5x0.5.nc"
    with nc.Dataset(inpath, mode='r') as f:
        #print(f.variables.keys())
        lat2 = (f.variables['lat'][:])
        lon2 = (f.variables['lon'][:])
        sm = (f.variables['sm'][:, :, :])
        sm_MG = MG(sm)
        sm = sm_MG.reshape(40, 12, 4, 30, 50)
        sm_gsl = sm[:, 4:9, :]
        sm_gsl = np.nanmean(sm_gsl, axis=1)
        sm_diff = sm_gsl[1:, ]-sm_gsl[:-1, ]
    return sm_diff    



# %% mask数组


def mask(x, data):
    sp = data.shape[0]
    lcc2 = np.array(lcc)
    
    np.place(lcc2, lcc2!=x, 1)
    np.place(lcc2, lcc2==x, 0)
    
    spei_ma = np.empty((sp, 4, 30, 50))
    spei_ma = np.ma.array(spei_ma)
    
    for i in range(sp):
        for l in range(4):
            a = data[i, l, :, :]
            a = ma.masked_array(a, mask=lcc2)
            spei_ma[i, l, :, :] = a
    
    return spei_ma.data

# %%


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
    ax.add_feature(provinces, linewidth=1, zorder=3)
    ax.add_feature(world, linewidth=1, zorder=3)

    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    ax.set_yticks(list(range(42, 55, 3)))
    ax.set_xticks(list(np.arange(102, 126, 5)))  # 需要显示的纬度

    ax.tick_params(labelsize=15)

    # 区域

    region = [100.25, 124.75, 40.25, 54.75]
    ax.set_extent(region, crs=ccrs.PlateCarree())

    # 设置网格点
    ylocs = list(range(42, 55, 3))
    xlocs = list(np.arange(102, 126, 5))
    lb = ax.gridlines(draw_labels=True, ylocs=ylocs, xlocs=xlocs, x_inline=False, y_inline=False,
                      linewidth=1, color='lightgray', linestyle='--', alpha=0.8)  # alpha是透明度

    lb.top_labels = False
    lb.bottom_labels = False
    lb.right_labels = False
    lb.left_labels = False
    lb.xlabel_style = {'size': 15}  # 修改经纬度字体大小
    lb.ylabel_style = {'size': 15}

    return ax


def Creat_figure(layer, data1, levels, cmap, lat, lon, title):
    fig = plt.figure(figsize=(15, 6), dpi=500)
    proj = ccrs.PlateCarree()

    fig.subplots_adjust(left=0.1, bottom=0.2, right=0.9,
                        top=0.9, wspace=0.1, hspace=0.05)

    lon, lat = np.meshgrid(lon, lat)

    for i in range(1, 6):
        ax = fig.add_subplot(2, 5, i, projection=proj)
        ax = make_map(ax)
        
        data = data1[i-1]
        data[lcc != 130] = np.nan
        norm = mpl.colors.BoundaryNorm(levels, cmap.N)  # 生成索引
        cs = ax.pcolormesh(lon, lat, data, norm=norm, cmap=cmap,
                           transform=ccrs.PlateCarree(), zorder=1)

        
        ax.set_title(f"{1998+i} minus {1998+i-1}", fontsize=12, loc="left")
        ax.tick_params(labelsize=12)
        if i==1:
            ax.set_ylabel(f"Layer{layer+1}", fontsize=12)
        if i != 1:
            ax.set_yticklabels([])

    # cbar_ax = fig.add_axes([0.95, 0.6, 0.01, 0.4])
    # cb = fig.colorbar(cs, cax=cbar_ax, orientation='vertical')
    # cb.ax.tick_params(labelsize=12)
    
    cbar_ax2 = fig.add_axes([0.3, 0.52, 0.4, 0.02])
    cb = fig.colorbar(cs, cax=cbar_ax2, orientation='horizontal')
    cb.ax.tick_params(labelsize=12)
    
    # plt.suptitle(f'{title}', fontsize=25)
    # plt.savefig(
    #     rf'E:/LAI4g/JPG_MG2/{title}.jpg', bbox_inches='tight')
    plt.show()


# %%
lcc = read_lcc()
sm = read_nc()
sm_mk = mask(130, sm)


# %% 1999-2003 annual
# yr = np.arange(1982, 2021)
# yr_p = yr[17:22]

# %%
sm_p = sm_mk[17:22, ]


def read_plot(data1, i):
    print('5 percentile is: ', np.nanpercentile(data1, 5))
    print('95 percentile is: ', np.nanpercentile(data1, 95), "\n")

    levels = np.arange(-24, 24.1, 2)
    cmap = cmaps.MPL_bwr_r[20:109]
    Creat_figure(i, data1, levels, cmap, lat, lon,
                  title=f"SM 1999-2003 Layer{i+1}")

for i in range(4):
    read_plot(sm_p[:, i], i)
