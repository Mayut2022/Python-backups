# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 09:26:24 2023

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

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import xarray as xr


plt.rcParams['font.sans-serif'] = ['times new roman']  # 指定默认字体

# %%


def read_nc():
    global lat, lon
    inpath = (r"E:/LAI4g/data_MG/LAI_82_20_MG_SPEI0.5x0.5.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        # print(f.variables['lai'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        lai = (f.variables['lai'][:, 4:9, :, :])
        lai1, lai2 = lai[:18, ], lai[18:, ]

        lai11 = lai1.reshape(18*5, 30, 50)
        lai22 = lai2.reshape(21*5, 30, 50)

    return lai11, lai22


def read_nc2():
    inpath = (r"E:/SPEI_base/data/spei03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        spei = (f.variables['spei'][972:, :, :])
        spei = spei.reshape(39, 12, 30, 50)
        spei_gsl = spei[:, 4:9, :].reshape(39*5, 30, 50)
        spei1, spei2 = spei_gsl[:90, :], spei_gsl[90:, :]
    return spei1, spei2

# %% mask数组, 统计干旱月份和对应的数据
# 筛选干旱与非干旱


def lai_drought(x, lai, spei):
    print(str(x))

    # 非干旱统计
    # 植被值
    laim1 = lai.copy()
    exec(f"laim1[{x}]=np.nan")
    laim_N = locals()['laim1']
    gNm = np.nanmean(laim_N, axis=0)
    # 月份统计
    laic1 = lai.copy()
    exec(f"laic1[{x}]=0")
    exec(f"laic1[~({x})]=1")
    laic_N = locals()['laic1']
    gNc = np.nansum(laic_N, axis=0)

    # 干旱统计
    # 植被值
    laim2 = lai.copy()
    exec(f"laim2[~({x})]=np.nan")
    laim_Y = locals()['laim2']
    gYm = np.nanmean(laim_Y, axis=0)
    # 月份统计
    laic2 = lai.copy()
    exec(f"laic2[~({x})]=0")
    exec(f"laic2[{x}]=1")
    laic_Y = locals()['laic2']
    gYc = np.nansum(laic_Y, axis=0)

    return gNm, gNc, gYm, gYc  # 依次为：非干旱的植被平均值，非干旱月数；干旱的植被平均值，干旱月数


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
    ax.add_feature(provinces, linewidth=1.2, zorder=3)
    ax.add_feature(world, linewidth=1.2, zorder=3)

    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    yticks = np.arange(42.5, 55, 5)
    xticks = np.arange(102.5, 130, 5)

    ax.set_yticks(yticks)
    ax.set_xticks(xticks)  # 需要显示的纬度
    ax.tick_params(labelsize=15)

    # 区域
    region = [100.25, 124.25, 40.25, 54.75]
    ax.set_extent(region, crs=ccrs.PlateCarree())

    # 设置网格点
    ylocs = list(yticks)
    xlocs = list(xticks)
    lb = ax.gridlines(draw_labels=True, ylocs=ylocs, xlocs=xlocs, x_inline=False, y_inline=False,
                      linewidth=1, color='lightgray', linestyle='--', alpha=0.8)  # alpha是透明度

    lb.top_labels = False
    lb.bottom_labels = False
    lb.right_labels = False
    lb.left_labels = False
    lb.xlabel_style = {'size': 15}  # 修改经纬度字体大小
    lb.ylabel_style = {'size': 15}

    return ax


def Creat_figure(data1, data2, data3, levels, cmap, lat, lon, title):
    fig = plt.figure(figsize=(16, 4), dpi=1080)
    proj = ccrs.PlateCarree()

    fig.subplots_adjust(left=0.12, bottom=0.12, right=0.88,
                        top=0.9, wspace=0.22, hspace=0.27)

    lon, lat = np.meshgrid(lon, lat)

    # title_str = ["(a) 1982-1999 Sum", "(b) 2000-2018 Sum", "(c) DIFF(b-a)"]
    title_str = ["(a) 1982-1999 Ave", "(b) 2000-2018 Ave", "(c) DIFF(b-a)"]
    for i in range(1, 4):
        ax = fig.add_subplot(1, 3, i, projection=proj)
        ax = make_map(ax)
        ax.set_title(f"{title_str[i-1]}", fontsize=15, loc="left")
        # lcc2 = np.ma.array(lcc, mask=lcc != 200)
        # ax.contourf(lon, lat, lcc2,
        #             transform=ccrs.PlateCarree(),
        #             colors="lightgray", zorder=2, corner_mask=False)
        if i == 1:
            cs = ax.contourf(lon, lat, data1,
                             levels=levels, transform=ccrs.PlateCarree(),
                             cmap=cmap, extend="both", zorder=1, corner_mask=False)
        elif i == 2:
            cs = ax.contourf(lon, lat, data2,
                             levels=levels, transform=ccrs.PlateCarree(),
                             cmap=cmap, extend="both", zorder=1, corner_mask=False)
        else:
            # levels2 = np.arange(0, 41.1, 2)
            # cmap2 = cmaps.BlueWhiteOrangeRed[127:225]  # 共254颜色
            levels2 = np.arange(-1.2, 1.21, 0.1)
            cmap2 = cmaps.MPL_BrBG[20:109]
            cs2 = ax.contourf(lon, lat, data3,
                              levels=levels2, transform=ccrs.PlateCarree(),
                              cmap=cmap2, extend="both", zorder=1, corner_mask=False)

    ######
    cbar_ax = fig.add_axes([0.23, 0.06, 0.3, 0.04])
    cb = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
    cb.ax.tick_params(labelsize=15)
    cb.set_label('Units: m2/m2', fontsize=15)
    ######
    cbar_ax = fig.add_axes([0.675, 0.06, 0.2, 0.04])
    cb = fig.colorbar(cs2, cax=cbar_ax, orientation='horizontal')
    cb.ax.tick_params(labelsize=15)
    cb.set_label('Units: m2/m2', fontsize=15)

    plt.suptitle(f'{title}', fontsize=20)
    plt.savefig(
        rf'E:/LAI4g/JPG_MG/Comparison/{title}.jpg', bbox_inches='tight')
    plt.show()


# %%
def PLOT(data1, data2, lev):
    data3 = data2-data1
    # levels = np.arange(20, 71, 5)
    levels = np.arange(10, 61, 5)
    cmap = cmaps.MPL_YlOrRd

    lev2 = lev.split('<')[1]
    title = rf"Drought Month (SPEI under {lev2})"
    print("\n", title)

    print('5 percentile is: ', np.nanpercentile(data1, 5))
    print('95 percentile is: ', np.nanpercentile(data2, 95), "\n")

    print("DIFF")
    print('5 percentile is: ', np.nanpercentile(data3, 5))
    print('95 percentile is: ', np.nanpercentile(data3, 95))

    # Creat_figure(data1, data2, data3, levels, cmap, lat, lon, title)


# %%
def PLOT2(data1, data2, lev):
    data3 = data2-data1
    levels = np.arange(0, 3.51, 0.2)
    cmap = "YlGn"

    lev2 = lev.split('<')[1]

    title = rf"{name} Drought LAI"

    print("\n", title)

    print('5 percentile is: ', np.nanpercentile(data2, 5))
    print('95 percentile is: ', np.nanpercentile(data2, 95), "\n")

    print("DIFF")
    print('5 percentile is: ', np.nanpercentile(data3, 5))
    print('95 percentile is: ', np.nanpercentile(data3, 95))

    Creat_figure(data1, data2, data3, levels, cmap, lat, lon, title)


# %%
lai1, lai2 = read_nc()
spei1, spei2 = read_nc2()

# %%
lev_d = [
    "(spei<-0.5)&(spei>=-1)", "(spei<-1)&(spei>=-1.5)", "(spei<-1.5)&(spei>=-2)", "(spei<-2)"]
lev_n = ["Mild", "Moderate", "Severe", "Extreme"]

for lev1, name in zip(lev_d[:], lev_n[:]):
    _, _, gYm1, _ = lai_drought(lev1, lai1, spei1)
    _, _, gYm2, _ = lai_drought(lev1, lai2, spei2)
    PLOT2(gYm1, gYm2, lev1)
