# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 12:20:04 2023
coefficient of variability
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


import netCDF4 as nc

import matplotlib as mpl
import numpy as np
from scipy.stats import pearsonr

plt.rcParams['font.sans-serif'] = ['times new roman']  # 指定默认字体

import warnings
warnings.filterwarnings("ignore")

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
    lai1, lai2 = lai_gs[:18, ], lai_gs[18:, ]
    return lai1, lai2


def read_nc2():
    inpath = (r"E:/SPEI_base/data/spei03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        spei = (f.variables['spei'][960:, :, :])
        spei = spei.reshape(40, 12, 30, 50)
        spei_gs = np.nanmean(spei[:, 4:9, :, :], axis=1)
    spei1, spei2 = spei_gs[:19, ], spei_gs[19:, ]
    return spei1, spei2


# %% 变率系数
### cv = 100*(std/Xave)
def coff_var(data):
    std = np.std(data, axis=0)
    Xave = np.nanmean(data, axis=0)
    cv = 10*(std/Xave)
    return cv

# %%


def CreatMap(data1, lon, lat, levels, cmap, title):
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
    ax.add_feature(provinces, linewidth=0.8, zorder=3)
    ax.add_feature(world, linewidth=0.8, zorder=2)

    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    yticks = np.arange(42.5, 55, 5)
    xticks = np.arange(102.5, 125, 5)
    ax.set_yticks(yticks)
    ax.set_xticks(xticks)  # 需要显示的纬度
    ax.tick_params(labelsize=12)

    # 区域
    region = [100.25, 124.75, 40.25, 54.75]
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
    # cs = ax.contourf(lon, lat, data1,
    #                  levels=levels, transform=ccrs.PlateCarree(),
    #                  cmap=cmap, extend="both", zorder=1, corner_mask=False)
    norm = mpl.colors.BoundaryNorm(levels, cmap.N)  # 生成索引
    cs = ax.pcolormesh(lon, lat, data1, norm=norm, cmap=cmap,
                       transform=ccrs.PlateCarree(), zorder=1)


    lcc2 = np.ma.array(lcc, mask=lcc != 200)
    ax.contourf(lon, lat, lcc2,
                transform=ccrs.PlateCarree(),
                colors="lightgray", zorder=2, corner_mask=False)
    ax.set_title(f"{title}", fontsize=20, loc="left")

    cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.03])
    cb = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
    #cb.set_label('units: mm/month', fontsize=15)
    cb.ax.tick_params(labelsize=12)

    plt.savefig(rf'E:/LAI4g/JPG_MG/{title}.jpg',
                bbox_inches='tight')
    plt.show()


# %%
lcc = read_lcc()
lai1, lai2 = read_nc()
spei1, spei2 = read_nc2()

lai1_cv, lai2_cv = coff_var(lai1), coff_var(lai2)
spei1_cv, spei2_cv = coff_var(spei1), coff_var(spei2)

lai_cv_diff = lai2_cv-lai1_cv
spei_cv_diff = spei2_cv-spei1_cv

# %%
def plot(data1):
    print("SPEI")
    print('5 percentile is: ', np.nanpercentile(data1, 5))
    print('95 percentile is: ', np.nanpercentile(data1, 95), "\n")


    levels = np.arange(-200, 201, 10)
    cmap = cmaps.BlueWhiteOrangeRed_r[20:225]
    title = r"CV DIFF SPEI"
    CreatMap(data1, lon, lat, levels, cmap, title)
    
def plot2(data1):
    print("LAI")
    print('5 percentile is: ', np.nanpercentile(data1, 5))
    print('95 percentile is: ', np.nanpercentile(data1, 95), "\n")
    

    levels = np.arange(-2, 2.1, 0.2)
    cmap = cmaps.MPL_BrBG[20:109]
    title = r"CV DIFF LAI"
    CreatMap(data1, lon, lat, levels, cmap, title)



# %%
plot(spei_cv_diff)
plot2(lai_cv_diff)
