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
import matplotlib as mpl
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
    inpath = (r"E:/LAI4g/data_MG/LAI_Detrend_82_20_MG_SPEI0.5x0.5.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        # print(f.variables['lai'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        lai = (f.variables['lai'][:, 4:9, :, :])
        lai_gs = np.nanmean(lai, axis=1)
        lai1, lai2 = lai_gs[:18], lai_gs[18:]
    return lai_gs, lai1, lai2


def read_nc2():
    inpath = (r"E:/SPEI_base/data/spei03_MG_detrend.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        # print(f.variables['spei'])
        spei = (f.variables['spei'][:])
        spei_gsl = spei[4:9, 1:, :]
        spei_gsl = np.nanmean(spei_gsl, axis=0)
        spei1, spei2 = spei_gsl[:18], spei_gsl[18:]
    return spei_gsl, spei1, spei2

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
    ax.add_feature(provinces, linewidth=1.2, zorder=4)
    ax.add_feature(world, linewidth=1.2, zorder=4)

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
    # lb = ax.gridlines(draw_labels=True, ylocs=ylocs, xlocs=xlocs, x_inline=False, y_inline=False,
    #                   linewidth=1, color='lightgray', linestyle='--', alpha=0.8)  # alpha是透明度

    # lb.top_labels = False
    # lb.bottom_labels = False
    # lb.right_labels = False
    # lb.left_labels = False
    # lb.xlabel_style = {'size': 15}  # 修改经纬度字体大小
    # lb.ylabel_style = {'size': 15}

    return ax


def Creat_figure(r, p, r1, p1, r2, p2, levels, cmap, lat, lon, title):
    fig = plt.figure(figsize=(16, 4), dpi=1080)
    proj = ccrs.PlateCarree()

    fig.subplots_adjust(left=0.12, bottom=0.12, right=0.88,
                        top=0.9, wspace=0.15, hspace=0.27)

    lon, lat = np.meshgrid(lon, lat)
    title_str = ["(a)1982-1999", "(b)2000-2020", "(c)1982-2020"]
    for i in range(1, 4):
        ax = fig.add_subplot(1, 3, i, projection=proj)
        ax = make_map(ax)
        # ax.set_title(f"{title_str[i-1]}", fontsize=15, loc="left")
        ax.text(100.5, 53.5, f"{title_str[i-1]}",
                transform=ccrs.PlateCarree(), fontsize=15)
        lcc2 = np.ma.array(lcc, mask=lcc == 130)
        ax.contourf(lon, lat, lcc2,
                    transform=ccrs.PlateCarree(),
                    colors="white", zorder=3, corner_mask=False)
        
        if i == 1:
            cs = ax.contourf(lon, lat, r1,
                              levels=levels, transform=ccrs.PlateCarree(),
                              cmap=cmap, extend="both", zorder=1)
            ax.contourf(lon, lat, p1,
                        levels=[0, 0.05, 1], transform=ccrs.PlateCarree(),
                        hatches=['..', None], colors="none", zorder=2)
        elif i == 2:
            cs = ax.contourf(lon, lat, r2,
                              levels=levels, transform=ccrs.PlateCarree(),
                              cmap=cmap, extend="both", zorder=1)
            ax.contourf(lon, lat, p2,
                        levels=[0, 0.05, 1], transform=ccrs.PlateCarree(),
                        hatches=['..', None], colors="none", zorder=2)
        else:
            cs = ax.contourf(lon, lat, r,
                              levels=levels, transform=ccrs.PlateCarree(),
                              cmap=cmap, extend="both", zorder=1)

            ax.contourf(lon, lat, p,
                        levels=[0, 0.05, 1], transform=ccrs.PlateCarree(),
                        hatches=['..', None], colors="none", zorder=2)

        if i == 2 or i == 3:
            ax.set_yticklabels([])

        ax.text(100.5, 53.5, f"{title_str[i-1]}",
                transform=ccrs.PlateCarree(), fontsize=15)

    ######
    cbar_ax = fig.add_axes([0.3, 0.06, 0.4, 0.04])
    cb = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
    cb.ax.tick_params(labelsize=15)

    # plt.suptitle(f'{title}', fontsize=20)
    plt.savefig(
        rf'E:/LAI4g/JPG_MG/{title}.jpg', bbox_inches='tight')
    plt.savefig(
        rf'F:/000000 论文图表最终版/{title}.jpg', bbox_inches='tight')
    plt.show()
    
    
# %%
lcc = read_lcc()
lai, lai1, lai2 = read_nc()
spei, spei1, spei2 = read_nc2()

r, p = corr(spei, lai)
r1, p1 = corr(spei1, lai1)
r2, p2 = corr(spei2, lai2)


#%%

def plot(r, p, r1, p1, r2, p2):
    print('5 percentile is: ', np.nanpercentile(r2, 5))
    print('95 percentile is: ', np.nanpercentile(r2, 95), "\n")

    levels = np.arange(0.2, 0.76, 0.05)
    
    cmap = cmaps.BlueWhiteOrangeRed[162:]
    title = r"SPEI-LAI CORR (Detrend)"
   
    Creat_figure(r, p, r1, p1, r2, p2, levels, cmap, lat, lon, title)

#%%

plot(r, p, r1, p1, r2, p2)
