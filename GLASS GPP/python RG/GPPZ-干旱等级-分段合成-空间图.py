# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 20:35:19 2023

@author: MaYutong
"""
import warnings
import cmaps
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.io.shapereader import Reader, natural_earth

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd
import xarray as xr

plt.rcParams['font.sans-serif'] = ['times new roman']  # 指定默认字体
warnings.filterwarnings("ignore")
# %%


# def read_nc():
#     global lat, lon
#     inpath = (r"E:/GLASS-GPP/MG detrend Zscore/MG_GPP_Zscore_SPEI_0.5x0.5.nc")
#     with nc.Dataset(inpath) as f:
#         # print(f.variables.keys())
#         # print(f.variables['GPP_Z'])
#         lat = (f.variables['lat'][:])
#         lon = (f.variables['lon'][:])
#         gpp = (f.variables['GPP_Z'][:, 4:9, :])
#         gpp1, gpp2 = gpp[:18, ], gpp[18:, ]

#         gpp11 = gpp1.reshape(18*5, 30, 50)
#         gpp22 = gpp2.reshape(19*5, 30, 50)

#     return gpp11, gpp22


# def read_nc2():
#     inpath = (r"E:/SPEI_base/data/spei03_MG.nc")
#     with nc.Dataset(inpath) as f:
#         # print(f.variables.keys())
#         spei = (f.variables['spei'][972:-24, :])
#         spei = spei.reshape(37, 12, 30, 50)
#         spei_gsl = spei[:, 4:9, :].reshape(37*5, 30, 50)
#         spei1, spei2 = spei_gsl[:90, :], spei_gsl[90:, :]
#     return spei1, spei2

# %% Summer 


def read_nc():
    global lat, lon
    inpath = (r"E:/GLASS-GPP/MG detrend Zscore/MG_GPP_Zscore_SPEI_0.5x0.5.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        # print(f.variables['GPP_Z'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        gpp = (f.variables['GPP_Z'][:, 5:8, :])
        gpp1, gpp2 = gpp[:18, ], gpp[18:, ]

        gpp11 = gpp1.reshape(18*3, 30, 50)
        gpp22 = gpp2.reshape(19*3, 30, 50)

    return gpp11, gpp22


def read_nc2():
    inpath = (r"E:/SPEI_base/data/spei03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        spei = (f.variables['spei'][972:-24, :])
        spei = spei.reshape(37, 12, 30, 50)
        spei_gsl = spei[:, 5:8, :].reshape(37*3, 30, 50)
        spei1, spei2 = spei_gsl[:54, :], spei_gsl[54:, :]
    return spei1, spei2


# %% mask数组, 统计干旱月份和对应的数据
# 筛选干旱与非干旱

def gpp_drought(x, gpp, spei):
    print(str(x))

    # 非干旱统计
    # 植被值
    gppm1 = gpp.copy()
    exec(f"gppm1[{x}]=np.nan")
    gppm_N = locals()['gppm1']
    gNm = np.nanmean(gppm_N, axis=0)
    # 月份统计
    gppc1 = gpp.copy()
    exec(f"gppc1[{x}]=0")
    exec(f"gppc1[~({x})]=1")
    gppc_N = locals()['gppc1']
    gNc = np.nansum(gppc_N, axis=0)

    # 干旱统计
    # 植被值
    gppm2 = gpp.copy()
    exec(f"gppm2[~({x})]=np.nan")
    gppm_Y = locals()['gppm2']
    gYm = np.nanmean(gppm_Y, axis=0)
    # 月份统计
    gppc2 = gpp.copy()
    exec(f"gppc2[~({x})]=0")
    exec(f"gppc2[{x}]=1")
    gppc_Y = locals()['gppc2']
    gYc = np.nansum(gppc_Y, axis=0)

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
            levels2 = np.arange(-1, 1.1, 0.1)  # 前三等级
            # levels2 = np.arange(-2.5, 2.51, 0.2) #Extreme等级
            cmap2 = cmaps.MPL_BrBG[20:109]
            # levels2 = np.arange(-20, 21.1, 2)
            # cmap2 = cmaps.BlueWhiteOrangeRed[30:225]  # 共254颜色
            cs2 = ax.contourf(lon, lat, data3,
                              levels=levels2, transform=ccrs.PlateCarree(),
                              cmap=cmap2, extend="both", zorder=1, corner_mask=False)

    ######
    cbar_ax = fig.add_axes([0.23, 0.06, 0.3, 0.04])
    cb = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
    cb.ax.tick_params(labelsize=15)
    # cb.set_label('Units: g C/m2/month', fontsize=15)
    ######
    cbar_ax = fig.add_axes([0.675, 0.06, 0.2, 0.04])
    cb = fig.colorbar(cs2, cax=cbar_ax, orientation='horizontal')
    cb.ax.tick_params(labelsize=15)
    # cb.set_label('Units: g C/m2/month', fontsize=15)

    plt.suptitle(f'{title}', fontsize=20)
    plt.savefig(
        rf'E:/GLASS-GPP/JPG MG/Comparison/{title}.jpg', bbox_inches='tight')
    plt.show()


# %%
def PLOT(data1, data2, lev):
    data3 = data2-data1
    # levels = np.arange(20, 71, 5)
    levels = np.arange(0, 25, 2)
    cmap = cmaps.MPL_YlOrRd

    title = rf"Drought Month ({name})"
    print("\n", title)

    print('5 percentile is: ', np.nanpercentile(data1, 5))
    print('95 percentile is: ', np.nanpercentile(data2, 95), "\n")

    print("DIFF")
    print('5 percentile is: ', np.nanpercentile(data3, 5))
    print('95 percentile is: ', np.nanpercentile(data3, 95), "\n")

    Creat_figure(data1, data2, data3, levels, cmap, lat, lon, title)


# %%
def PLOT2(data1, data2, name):
    data3 = data2-data1
    # levels = np.arange(-1, 1.1, 0.1)
    levels = np.arange(-0.9, 0.91, 0.1)  # extreme
    cmap = cmaps.MPL_BrBG[20:109]

    title = rf"Drought GPPZ ({name})"
    title = rf"Summer Drought GPPZ ({name})"
    print("\n", title)

    print('5 percentile is: ', np.nanpercentile(data1, 5))
    print('95 percentile is: ', np.nanpercentile(data1, 95), "\n")

    print("DIFF")
    print('5 percentile is: ', np.nanpercentile(data3, 5))
    print('95 percentile is: ', np.nanpercentile(data3, 95), "\n")

    Creat_figure(data1, data2, data3, levels, cmap, lat, lon, title)


# %%
gpp1, gpp2 = read_nc()
spei1, spei2 = read_nc2()

# gpp_m1 = gpp_count("(spei<-0.5)&(spei>=-1)", gpp1, spei1)
# %%
# lev_d = [
#     "(spei<-0.5)&(spei>=-1)", "(spei<-1)&(spei>=-1.5)", "(spei<-1.5)&(spei>=-2)", "(spei<-2)"]
# lev_n = ["Mild", "Moderate", "Severe", "Extreme"]

# for lev1, name in zip(lev_d[:], lev_n[:]):
#     _, _, gYm1, gYc1 = gpp_drought(lev1, gpp1, spei1)
#     _, _, gYm2, gYc2 = gpp_drought(lev1, gpp2, spei2)
#     # PLOT2(gYm1, gYm2, name)
#     PLOT(gYc1, gYc2, lev1)


# %%
# PLOT(gYc1, gYc2, lev1)
# PLOT2(gYm1, gYm2, lev1)

# %%
lev_d = ["(spei<-0.5)&(spei>=-2)", "(spei<-2)"]
lev_n = ["MMS", "Extreme"]

for lev1, name in zip(lev_d[:1], lev_n[:1]):
    _, _, gYm1, gYc1 = gpp_drought(lev1, gpp1, spei1)
    _, _, gYm2, gYc2 = gpp_drought(lev1, gpp2, spei2)
    PLOT2(gYm1, gYm2, name)
    # PLOT(gYc1, gYc2, lev1)
