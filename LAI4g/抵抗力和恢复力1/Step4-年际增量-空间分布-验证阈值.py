# -*- coding: utf-8 -*-
"""
Created on Wed May 10 18:25:50 2023

@author: MaYutong
"""

import warnings
import cmaps
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.io.shapereader import Reader, natural_earth

import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

import pandas as pd
import xarray as xr


plt.rcParams['font.sans-serif'] = ['times new roman']  # 指定默认字体
warnings.filterwarnings("ignore")

# %%
# yr = np.arange(1983, 2021)
# print(yr[17])

# %%


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


def read_LAI():
    global lat, lon
    inpath = (r"E:/LAI4g/data_MG/LAI_82_20_MG_SPEI0.5x0.5.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        # print(f.variables['lai'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        lai = (f.variables['lai'][:, 4:9, :, :])
        lai_gs = np.nanmean(lai, axis=1)
        lai_diff = lai_gs[1:, ]-lai_gs[:-1, ]
        lai1, lai2 = lai_diff[:17], lai_diff[17:]
    return lai_diff, lai1, lai2

# %%


def read_spei_1():
    inpath = (r"E:/SPEI_base/data/spei03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        spei = (f.variables['spei'][960:, :, :])
        spei = spei.reshape(40, 12, 30, 50)
        spei_gsl = spei[:, 4:9, :]
        spei_gs = np.nanmean(spei_gsl, axis=1)
        spei_diff = spei_gs[1:, ]-spei_gs[:-1, ]
        spei1, spei2 = spei_diff[1:18], spei_diff[18:]
    return spei_diff[1:], spei1, spei2


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
    # ax.add_feature(cfeature.RIVERS.with_scale('50m'), linewidth=1, zorder=2)

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
    xticks = np.arange(102, 130, 5)

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
                        top=0.9, wspace=0.15, hspace=0.27)

    lon, lat = np.meshgrid(lon, lat)

    # title_str = ["(a) 1982-1999 Sum", "(b) 2000-2018 Sum", "(c) DIFF(b-a)"]
    title_str = ["(a) SPEI<-1.25", "(b) SPEI>1.25", "(c) DIFF(b-a)"]
    for i in range(1, 4):
        ax = fig.add_subplot(1, 3, i, projection=proj)
        ax = make_map(ax)
        ax.set_title(f"{title_str[i-1]}", fontsize=15, loc="left")
        data1_mk = np.ma.array(data1, mask=lcc != 130)
        data2_mk = np.ma.array(data2, mask=lcc != 130)
        data3_mk = np.ma.array(data3, mask=lcc != 130)
        norm = mpl.colors.BoundaryNorm(levels, cmap.N)
        if i == 1:
            cs = ax.pcolormesh(lon, lat, data1_mk, norm=norm, cmap=cmap,
                       transform=ccrs.PlateCarree(), zorder=1)
        elif i == 2:
            cs = ax.pcolormesh(lon, lat, data2_mk, norm=norm, cmap=cmap,
                       transform=ccrs.PlateCarree(), zorder=1)
        else:
            # levels2 = np.arange(0, 41.1, 2)
            # cmap2 = cmaps.BlueWhiteOrangeRed[127:225]  # 共254颜色
            levels2 = np.arange(-0.25, 0.26, 0.05)
            # cmap2 = cmaps.BlueWhiteOrangeRed_r
            cmap2 = cmaps.MPL_bwr_r
            norm2 = mpl.colors.BoundaryNorm(levels2, cmap2.N)
            cs2 = ax.pcolormesh(lon, lat, data3_mk, norm=norm2, cmap=cmap2,
                       transform=ccrs.PlateCarree(), zorder=1)
            
        if i==2 or i==3:
            ax.set_yticklabels([])

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
        rf'E:/LAI4g/python_MG/抵抗力和恢复力1/JPG3/{title}.jpg', bbox_inches='tight')
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
def PLOT2(data1, data2, title):
    data3 = data2-abs(data1)
    levels = np.arange(-0.4, 0.41, 0.05)
    # cmap = cmaps.BlueWhiteOrangeRed_r
    cmap = cmaps.MPL_BrBG

    print("\n", title)
    
    print('5 percentile is: ', np.nanpercentile(data1, 5))
    print('95 percentile is: ', np.nanpercentile(data1, 95), "\n")

    print('5 percentile is: ', np.nanpercentile(data2, 5))
    print('95 percentile is: ', np.nanpercentile(data2, 95), "\n")

    print("DIFF")
    print('5 percentile is: ', np.nanpercentile(data3, 5))
    print('95 percentile is: ', np.nanpercentile(data3, 95))

    Creat_figure(data1, data2, data3, levels, cmap, lat, lon, title)


# %%
lcc = read_lcc()
_, _, lai2 = read_LAI()
_, _, spei2 = read_spei_1()

# %%
lev_d = [
    "(spei<0)&(spei>=-1.25)", "(spei<-1.25)", "(spei<1.25)&(spei>0)", "(spei>1.25)"]
lev_n = ["Dry", "Extreme Dry", "Wet", "Extreme Wet"]

for i, lev, name in zip(range(1, 5), lev_d[:], lev_n[:]):
    exec(f"_, _, gYm{i}, _ = lai_drought(lev, lai2, spei2)")


# %%
PLOT2(gYm2, gYm4, "Significant Change 2000-2020")
