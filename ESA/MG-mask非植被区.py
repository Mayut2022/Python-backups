# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 11:26:10 2023

@author: MaYutong
"""
import cmaps
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.io.shapereader import Reader, natural_earth

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np

import xarray as xr

plt.rcParams['font.sans-serif'] = ['times new roman']  # 指定默认字体
# %%


def read_lcc():
    global lat, lon
    inpath = (r"E:\ESA\ESA_LUC_2000_SPEI03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        # print(f.variables['lccs_class'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        lcc = (f.variables['lcc'][:])

    return lcc

# %%


def CreatMap(data1, data2, lon, lat, levels, cmap, title):
    fig = plt.figure(figsize=(10, 6), dpi=1080)
    proj = ccrs.PlateCarree()
    ax = fig.add_subplot(111, projection=proj)
    fig.subplots_adjust(left=0.1, bottom=0.2, right=0.9,
                        top=0.9, wspace=None, hspace=None)

    # 打开边框
    ax.spines['geo'].set_visible(True)

    # 设置cfeature自带底图
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1)
    # ax.add_feature(cfeature.RIVERS.with_scale('50m'), linewidth=1, zorder=2)

    # 设置shp文件
    shp_path1 = r'E:/SHP/gadm36_CHN_shp/gadm36_CHN_1.shp'
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

    yticks = np.arange(42.5, 55, 5)
    xticks = np.arange(102.5, 130, 5)

    # yticks = np.arange(-90, 90, 30)
    # xticks = np.arange(-180, 181, 60)

    ax.set_yticks(yticks)
    ax.set_xticks(xticks)  # 需要显示的纬度
    ax.tick_params(labelsize=15)

    # 区域
    region = [100.25, 124.25, 40.25, 54.75]
    # region = [-180, 180, -90, 90]
    ax.set_extent(region, crs=ccrs.PlateCarree())

    # 设置网格点
    ylocs = yticks
    xlocs = xticks
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
                      cmap=cmap, extend="both", zorder=1, corner_mask=False)
    
    cs2 = ax.contourf(lon, lat, data2,
                      transform=ccrs.PlateCarree(),
                      colors="lightgray", zorder=1, corner_mask=False)
    
    cs3 = ax.contourf(lon, lat, lcc3,
                      transform=ccrs.PlateCarree(),
                      colors="yellow", zorder=1, corner_mask=False)

    cbar_ax = fig.add_axes([0.25, 0.1, 0.5, 0.03])
    cb = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')

    plt.suptitle(f'{title}', fontsize=20)
    plt.savefig(
        rf'E:/ESA/JPG/{title}.jpg', bbox_inches='tight')
    plt.show()

# %%


def plot(data1, data2):

    levels = np.arange(0, 210, 20)
    cmap = "YlGn"
    title = "MG LCC Mask Test"

    CreatMap(data1, data2, lon, lat, levels, cmap, title)




# %%
lcc = read_lcc()

lcc2 = np.ma.array(lcc, mask=lcc!=200)
lcc3 = np.ma.array(lcc, mask=lcc!=130)
plot(lcc, lcc2)
