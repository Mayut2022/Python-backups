# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 11:14:13 2022
abcdef
a\b是Rmax平均，c是区域平均变化；
d/e是14年中所有RmaxScale所占的比例，f是差值


@author: MaYutong
"""


from collections import Counter
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
import matplotlib.dates as mdate
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
import xarray as xr
import xlsxwriter

import seaborn as sns

from scipy.stats import linregress
from scipy.stats import pearsonr

# %%


def read_nc(inpath):
    global lat, lon
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())

        r = (f.variables['Rmax'][:])
        r_s = (f.variables['Rmax_scale'][:])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])

        r_c = r.copy()
        r_c[r_c >= 0] = 1
        r_c[r_c < 0] = -1

    return r, r_s, r_c

# %%


def read_nc3():
    global lcc
    # global a1, a2, o1, o2
    inpath = (r"E:/ESA/ESA_LUC_2000_SPEI03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        # print(f.variables['lccs_class'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        lcc = (f.variables['lcc'][:])

# %%


def make_map(ax):
    # 打开边框
    ax.spines['geo'].set_visible(True)

    # 设置cfeature自带底图
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1)
    ax.add_feature(cfeature.RIVERS.with_scale('50m'), linewidth=1, zorder=2)

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

    ax.set_yticks(list(range(42, 55, 3)))
    ax.set_xticks(list(np.arange(102.5, 126, 5)))  # 需要显示的纬度

    ax.tick_params(labelsize=15)

    # 区域

    region = [100.25, 124.75, 40.25, 54.75]
    ax.set_extent(region, crs=ccrs.PlateCarree())

    # 设置网格点
    ylocs = list(range(42, 55, 3))
    xlocs = list(np.arange(102.5, 126, 5))
    lb = ax.gridlines(draw_labels=True, ylocs=ylocs, xlocs=xlocs, x_inline=False, y_inline=False,
                      linewidth=1, color='lightgray', linestyle='--', alpha=0.8)  # alpha是透明度

    lb.top_labels = False
    lb.bottom_labels = False
    lb.right_labels = False
    lb.left_labels = False
    lb.xlabel_style = {'size': 15}  # 修改经纬度字体大小
    lb.ylabel_style = {'size': 15}

    return ax

####################################################


def Creat_figure(data1, data2, levels, cmap, lat, lon, title):
    fig = plt.figure(figsize=(12, 4), dpi=500)
    proj = ccrs.PlateCarree()

    fig.subplots_adjust(left=0.1, bottom=0.15, right=0.9,
                        top=0.9, wspace=0.2, hspace=0.2)

    lon, lat = np.meshgrid(lon, lat)

    yr_str = ["82-99 ave", "00-18 ave"]
    for i in range(1, 3):
        if i != 3:
            ax = fig.add_subplot(1, 2, i, projection=proj)
            ax = make_map(ax)
            ax.set_title(f"Rmax {yr_str[i-1]}",
                         fontsize=15, loc="left", color="b")
            cs = eval(
                f'ax.contourf(lon, lat, data{i}, levels=levels, transform=ccrs.PlateCarree(), cmap=cmap, zorder=1)')
            lcc2 = np.ma.array(lcc, mask=lcc != 200)
            cs2 = ax.contourf(lon, lat, lcc2, transform=ccrs.PlateCarree(
            ), colors="lightgray", zorder=1, corner_mask=False)

    cbar_ax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
    cb = fig.colorbar(cs, cax=cbar_ax, orientation='vertical')
    # cb.set_label('Units: mm/month (CRU-GLEAM)', fontsize=15)
    cb.ax.tick_params(labelsize=15)

    plt.suptitle(f'{title}', fontsize=20)
    plt.savefig(
        rf'E:/SPEI_base/python MG/JPG/Rmax_Scale分析/{title}.jpg', bbox_inches='tight')
    plt.show()


# %%


def make_map(ax):
    # 打开边框
    ax.spines['geo'].set_visible(True)

    # 设置cfeature自带底图
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1)
    ax.add_feature(cfeature.RIVERS.with_scale('50m'), linewidth=1, zorder=2)

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

    ax.set_yticks(list(range(42, 55, 3)))
    ax.set_xticks(list(np.arange(102.5, 126, 5)))  # 需要显示的纬度

    ax.tick_params(labelsize=15)

    # 区域

    region = [100.25, 124.75, 40.25, 54.75]
    ax.set_extent(region, crs=ccrs.PlateCarree())

    # 设置网格点
    ylocs = list(range(42, 55, 3))
    xlocs = list(np.arange(102.5, 126, 5))
    lb = ax.gridlines(draw_labels=True, ylocs=ylocs, xlocs=xlocs, x_inline=False, y_inline=False,
                      linewidth=1, color='lightgray', linestyle='--', alpha=0.8)  # alpha是透明度

    lb.top_labels = False
    lb.bottom_labels = False
    lb.right_labels = False
    lb.left_labels = False
    lb.xlabel_style = {'size': 15}  # 修改经纬度字体大小
    lb.ylabel_style = {'size': 15}

    return ax

####################################################


def Creat_figure(data1, data2, levels, cmap, lat, lon, title):
    fig = plt.figure(figsize=(12, 4), dpi=500)
    proj = ccrs.PlateCarree()

    fig.subplots_adjust(left=0.1, bottom=0.15, right=0.9,
                        top=0.9, wspace=0.2, hspace=0.2)

    lon, lat = np.meshgrid(lon, lat)

    yr_str = ["82-99 ave", "00-18 ave"]
    for i in range(1, 3):
        if i != 3:
            ax = fig.add_subplot(1, 2, i, projection=proj)
            ax = make_map(ax)
            ax.set_title(f"Rmax {yr_str[i-1]}",
                         fontsize=15, loc="left", color="b")
            cs = eval(
                f'ax.contourf(lon, lat, data{i}, levels=levels, transform=ccrs.PlateCarree(), cmap=cmap, extend="both", zorder=1)')
            lcc2 = np.ma.array(lcc, mask=lcc != 200)
            cs2 = ax.contourf(lon, lat, lcc2, transform=ccrs.PlateCarree(
            ), colors="lightgray", zorder=1, corner_mask=False)

    cbar_ax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
    cb = fig.colorbar(cs, cax=cbar_ax, orientation='vertical')
    # cb.set_label('Units: mm/month (CRU-GLEAM)', fontsize=15)
    cb.ax.tick_params(labelsize=15)

    plt.suptitle(f'{title}', fontsize=20)
    # plt.savefig(
    #     rf'E:/SPEI_base/python MG/JPG/Rmax_Scale分析/{title}.jpg', bbox_inches='tight')
    plt.show()


# %%
read_nc3()


# %% 画图


def read_plot(data1, data2, title):
    print('5 percentile is: ', np.nanpercentile(data1, 5))
    print('95 percentile is: ', np.nanpercentile(data1, 95), "\n")

    levels = [1, 3, 10, 12]
    cmap = "RdBu"
    Creat_figure(data1, data2, levels, cmap, lat, lon, title)


def most_data(data):
    data_cut = np.zeros((30, 50))
    for i in range(30):
        for j in range(50):
            if np.isnan(data[:, i, j]).any():
                data_cut[i, j] = np.nan
            else:
                a = data[:, i, j]
                b = np.array(a)
                b = b.astype('int64')
                count = np.bincount(b)
                most = np.argmax(count)
                data_cut[i, j] = most

    return data_cut


# %% 修改图，只将非植被区mask掉
inpath = rf"E:/SPEI_base/python MG/response time data/GPP_SPEI_MG_Summer.nc"
_, r_s, _ = read_nc(inpath)


r_s_1 = r_s[:14, :, :]
r_s_2 = r_s[14:, :, :]

r_s_1_plot = most_data(r_s_1)
r_s_2_plot = most_data(r_s_2)

read_plot(r_s_1_plot, r_s_2_plot,
          title=fr"GPP RmaxScale Comparison Summer")
