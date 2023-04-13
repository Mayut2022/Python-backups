# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 12:36:46 2023
生长季平均
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

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

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

    return lai_gs


def read_nc2():
    inpath = (r"E:/SPEI_base/data/spei03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        spei = (f.variables['spei'][972:, :, :])
        spei = spei.reshape(39, 12, 30, 50)
        spei_gsl = np.nanmean(spei[:, 4:9, :, :], axis=1)

    return spei_gsl

# %%


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

def corr2(data1, data2):
    r = np.zeros((30, 50))
    p = np.zeros((30, 50))
    for i in range(30):
        for j in range(50):
            if np.isnan(data1[:, i, j]).any():
                r[i, j], p[i, j] = np.nan, np.nan
            else:
                r[i, j], p[i, j] = pearsonr(data1[:, i, j], data2)

    return r, p

# %%


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
    cs = ax.contourf(lon, lat, data1,
                     levels=levels, transform=ccrs.PlateCarree(),
                     cmap=cmap, extend="both", zorder=1, corner_mask=False)

    cs2 = ax.contourf(lon, lat, data2,
                      levels=[0, 0.05, 1], transform=ccrs.PlateCarree(),
                      hatches=['.', None], colors="none", zorder=3)
    # lcc2 = np.ma.array(lcc, mask=lcc != 200)
    # ax.contourf(lon, lat, lcc2,
    #             transform=ccrs.PlateCarree(),
    #             colors="lightgray", zorder=2, corner_mask=False)

    cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.03])
    cb = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
    #cb.set_label('units: mm/month', fontsize=15)
    cb.ax.tick_params(labelsize=12)
    plt.suptitle(f"{title}", fontsize=20)
    plt.savefig(rf'E:/LAI4g/JPG_MG/{title}.jpg',
                bbox_inches='tight')
    plt.show()


# %%
lcc = read_lcc()
lai, spei = read_nc(), read_nc2()
r, p = corr(spei, lai)


def plot(data1, data2, i):
    print('5 percentile is: ', np.nanpercentile(data1, 5))
    print('95 percentile is: ', np.nanpercentile(data1, 95), "\n")

    levels = np.arange(-0.7, 0.71, 0.05)
    cmap = cmaps.MPL_bwr_r[15:113]
    title = r"SPEI-LAI CORR GSL Mean"
    # title = rf"SPEI-PC{i} CORR GSL Mean"
    CreatMap(data1, data2, lon, lat, levels, cmap, title)


plot(r, p, 0)
# %% EOF 时间序列相关性
# pc = np.load("LAI_EOF_PC.npy")
# for i in range(3):
#     exec(f"r{i+1}, p{i+1} = corr2(lai, pc[:, i])")
#     exec(f"plot(r{i+1}, p{i+1}, i+1)")