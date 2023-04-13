# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 12:45:08 2023

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
from matplotlib import cm
import numpy as np
import numpy.ma as ma
import pandas as pd

import warnings
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['times new roman']  # 指定默认字体

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


def read_nc():
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

    return lai_diff

# %%


def read_nc2():
    inpath = (r"E:/SPEI_base/data/spei03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        spei = (f.variables['spei'][972:, :, :])
        spei = spei.reshape(39, 12, 30, 50)
        spei_gsl = spei[:, 4:9, :]
        spei_gsl = np.nanmean(spei_gsl, axis=1)  # 82-20
        spei_gsl2 = spei_gsl[1:, ]  # 83-20
        spei_diff = spei_gsl2-spei_gsl[:-1, ]

    return spei_diff


# %% mask数组


def mask(x, data):
    sp = data.shape[0]
    lcc2 = np.array(lcc)

    np.place(lcc2, lcc2 != x, 1)
    np.place(lcc2, lcc2 == x, 0)

    spei_ma = np.empty((sp, 30, 50))
    spei_ma = np.ma.array(spei_ma)

    for i in range(sp):
        a = data[i, :, :]
        a = ma.masked_array(a, mask=lcc2)
        spei_ma[i, :, :] = a

    spei_ma_ave = np.nanmean(spei_ma, axis=(1, 2))

    return spei_ma_ave

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


def Creat_figure(data1, levels, cmap, lat, lon, title):
    fig = plt.figure(figsize=(8, 6), dpi=500)
    proj = ccrs.PlateCarree()

    fig.subplots_adjust(left=0.1, bottom=0.2, right=0.9,
                        top=0.9, wspace=0.1, hspace=0.05)

    lon, lat = np.meshgrid(lon, lat)

    for i in range(1, 5):
        ax = fig.add_subplot(2, 2, i, projection=proj)
        ax = make_map(ax)
        cs = ax.contourf(lon, lat, data1[i-1],
                         levels=levels, transform=ccrs.PlateCarree(),
                         cmap=cmap, extend="both", zorder=1, corner_mask=False)
        lcc2 = np.ma.array(lcc, mask=lcc != 200)
        ax.contourf(lon, lat, lcc2,
                    transform=ccrs.PlateCarree(),
                    colors="lightgray", zorder=2, corner_mask=False)
        
        # lcc3 = np.ma.array(lcc, mask=lcc != 130)
        # ax.contourf(lon, lat, lcc3,
        #             transform=ccrs.PlateCarree(),
        #             colors="lightgray", zorder=2, corner_mask=False)

        ax.tick_params(labelsize=15)
        if i==1:
            ax.set_title(f"(a) Wet to Dry Before 1999", loc="left", fontsize=15)
            ax.set_xticklabels([])
        elif i==2:
            ax.set_title(f"(b) Wet to Dry After 1999", loc="left", fontsize=15)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        elif i==3:
            ax.set_title(f"(c) Dry to Wet Before 1999", loc="left", fontsize=15)
        elif i==4:
            ax.set_title(f"(d) Dry to Wet After 1999", loc="left", fontsize=15)
            ax.set_yticklabels([])


    cbar_ax = fig.add_axes([0.2, 0.1, 0.6, 0.02])
    cb = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
    cb.ax.tick_params(labelsize=15)

    # plt.suptitle(f'{title}', fontsize=25)
    plt.savefig(
        rf'E:/LAI4g/JPG_MG/{title}.jpg', bbox_inches='tight')
    plt.show()


# %%
lcc = read_lcc()
lai = read_nc()
lai_mk = mask(130, lai)


spei = read_nc2()
spei_mk = mask(130, spei)


# %%
year = np.arange(1983, 2021)

w2d = spei_mk < -0.8
yr_w2d = year[w2d]
print("Wet->Dry:", yr_w2d, "\n")

d2w = spei_mk > 0.8
yr_d2w = year[d2w]
print("Dry->Wet:", yr_d2w, "\n")

# %%
lai_w2d = lai[w2d, :, :]
lai_w2d1 = np.nanmean(lai_w2d[:3, ], axis=0)
lai_w2d2 = np.nanmean(lai_w2d[3:, ], axis=0)

lai_d2w = lai[d2w, :, :]
lai_d2w1 = np.nanmean(lai_d2w[:3, ], axis=0)
lai_d2w2 = np.nanmean(lai_d2w[3:, ], axis=0)



# %% 画图
def read_plot(data1):
    print('5 percentile is: ', np.nanpercentile(data1, 5))
    print('95 percentile is: ', np.nanpercentile(data1, 95), "\n")


    levels = np.arange(-0.2, 0.21, 0.01)
    cmap = cmaps.MPL_BrBG[20:109]
    Creat_figure(data1, levels, cmap, lat, lon, title=f"MG LAI Change Annual")
    
lai_all = [lai_w2d1, lai_w2d1, lai_d2w1, lai_d2w2]
read_plot(lai_all)