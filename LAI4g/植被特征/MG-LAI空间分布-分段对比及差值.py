# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 19:12:46 2023

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
from scipy import stats
import xarray as xr

plt.rcParams['font.sans-serif'] = ['times new roman']  # 指定默认字体

# %%


def read_nc():
    global lcc, lat, lon
    inpath = (r"E:\ESA\ESA_LUC_2000_SPEI03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        # print(f.variables['lccs_class'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        lcc = (f.variables['lcc'][:])

# %%

def read_nc2():
    global lat2, lon2
    inpath = (r"E:/LAI4g/data_MG/LAI_82_20_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        print(f.variables['lai'])
        lat2 = (f.variables['lat'][:])
        lon2 = (f.variables['lon'][:])
        lai = (f.variables['lai'][:, 4:9, :, :])
        lai_gsl = np.nanmean(lai, axis=1)

    return lai_gsl


# %%


def tTest(list1, list2):
    t = np.zeros((180, 300))
    pt = np.zeros((180, 300))

    for r in range(180):
        if r % 180 == 0:
            print(f"column {r} is done!")
        for c in range(300):
            a = list1[:, r, c]
            b = list2[:, r, c]
            if np.isnan(a).any() or np.isnan(b).any():
                t[r, c] = np.nan
                pt[r, c] = np.nan
            else:
                levene = stats.levene(a, b, center='median')
                if levene[1] < 0.05:
                    t[r, c] = np.nan
                    pt[r, c] = np.nan
                else:
                    tTest = stats.stats.ttest_ind(a, b, equal_var=True)
                    t[r, c] = tTest[0]
                    pt[r, c] = tTest[1]

    return t, pt


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


def Creat_figure(data1, data2, data3, data4, levels, cmap, lat, lon, title):
    fig = plt.figure(figsize=(16, 4), dpi=1080)
    proj = ccrs.PlateCarree()

    fig.subplots_adjust(left=0.12, bottom=0.12, right=0.88,
                        top=0.9, wspace=0.15, hspace=0.27)

    lon, lat = np.meshgrid(lon, lat)
    title_str = ["(d)1982-1999 Ave", "(e)2000-2020 Ave", "(f)DIFF(e-d)"]
    for i in range(1, 4):
        ax = fig.add_subplot(1, 3, i, projection=proj)
        ax = make_map(ax)
        # ax.set_title(f"{title_str[i-1]}", fontsize=15, loc="left")
        ax.text(100.5, 53.5, f"{title_str[i-1]}", transform=ccrs.PlateCarree(), fontsize=15)
        lcc2 = np.ma.array(lcc, mask=lcc == 130)
        ax.contourf(lon, lat, lcc2,
                    transform=ccrs.PlateCarree(),
                    colors="white", zorder=3, corner_mask=False)
        if i==1:
            cs = ax.contourf(lon2, lat2, data1,
                             levels=levels, transform=ccrs.PlateCarree(),
                             cmap=cmap, extend="both", zorder=1)
        elif i==2:
            cs = ax.contourf(lon2, lat2, data2,
                             levels=levels, transform=ccrs.PlateCarree(),
                             cmap=cmap, extend="both", zorder=1)
        else:
            levels2 = np.arange(-0.3, 0.31, 0.05)
            cmap2 = cmaps.MPL_BrBG[20:109]
            cs2 = ax.contourf(lon2, lat2, data3,
                             levels=levels2, transform=ccrs.PlateCarree(),
                             cmap=cmap2, extend="both", zorder=1)
            # data4 = np.ma.array(data4, mask=lcc != 130)
            cs3 = ax.contourf(lon2, lat2, data4,
                              levels=[0, 0.05, 1], transform=ccrs.PlateCarree(),
                              hatches=['..', None], colors="none", zorder=2)
            
        if i==2 or i==3:
            ax.set_yticklabels([])
            
        ax.text(100.5, 53.5, f"{title_str[i-1]}", transform=ccrs.PlateCarree(), fontsize=15)
        
    ######
    cbar_ax = fig.add_axes([0.22, 0.06, 0.3, 0.04])
    cb = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
    cb.ax.tick_params(labelsize=15)
    # cb.set_label('Units: m2/m2', fontsize=15)
    ######
    cbar_ax = fig.add_axes([0.66, 0.06, 0.2, 0.04])
    cb = fig.colorbar(cs2, cax=cbar_ax, orientation='horizontal')
    cb.ax.tick_params(labelsize=15)
    # cb.set_label('Units: m2/m2', fontsize=15)

    # plt.suptitle(f'{title}', fontsize=20)
    # plt.savefig(
    #     rf'E:/LAI4g/JPG_MG/{title}.jpg', bbox_inches='tight')
    plt.savefig(
        rf'F:/000000 论文图表最终版/{title}.jpg', bbox_inches='tight')
    plt.show()


# %%
def PLOT(data1, data2, data3, data4):
    

    print('5 percentile is: ', np.nanpercentile(data3, 5))
    print('95 percentile is: ', np.nanpercentile(data3, 95))

    levels = np.arange(0, 3.1, 0.5)
    cmap = "YlGn"
    title = r"MG LAI GSL Diff (82-99)(00-20)"

    Creat_figure(data1, data2, data3, data4, levels, cmap, lat, lon, title)


# %%


read_nc()
lai = read_nc2()

lai1 = lai[:19, :, :]  # 81-99
lai2 = lai[19:, :, :]  # 00-18

lai1_ave = np.nanmean(lai1, axis=0)
lai2_ave = np.nanmean(lai2, axis=0)
lai_diff = lai2_ave-lai1_ave

_, pt = tTest(lai1, lai2)
PLOT(lai1_ave, lai2_ave, lai_diff, pt)
