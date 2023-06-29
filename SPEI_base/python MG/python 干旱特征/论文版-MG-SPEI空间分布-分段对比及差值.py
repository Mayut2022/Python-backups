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
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
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
    inpath = (r"E:/SPEI_base/data/spei03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        spei = (f.variables['spei'][960:, :, :])
        spei = spei.reshape(40, 12, 30, 50)
        spei_gsl = spei[:, 4:9, :]
        spei_gsl = np.nanmean(spei_gsl, axis=1)  # 81-20
        return spei_gsl

# %%


def tTest(list1, list2):
    t = np.zeros((30, 50))
    pt = np.zeros((30, 50))

    for r in range(30):
        if r % 30 == 0:
            print(f"column {r} is done!")
        for c in range(50):
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
    # ax.add_feature(cfeature.LAKES.with_scale('50m'), linewidth=1, zorder=2)

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
    title_str = ["(a)1982-1999 Ave", "(b)2000-2020 Ave", "(c)DIFF(b-a)"]
    # title_str = ["(a)", "(b)", "(c)"]
    for i in range(1, 4):
        ax = fig.add_subplot(1, 3, i, projection=proj)
        ax = make_map(ax)
        # ax.set_title(f"{title_str[i-1]}", fontsize=15, loc="left")
        ax.text(100.5, 53.5, f"{title_str[i-1]}", transform=ccrs.PlateCarree(), fontsize=15)
        lcc2 = np.ma.array(lcc, mask=lcc == 130)
        ax.contourf(lon, lat, lcc2,
                    transform=ccrs.PlateCarree(),
                    colors="white", zorder=2, corner_mask=False)
        if i==1:
            cs = ax.contourf(lon, lat, data1,
                             levels=levels, transform=ccrs.PlateCarree(),
                             cmap=cmap, extend="both", zorder=1)
        elif i==2:
            cs = ax.contourf(lon, lat, data2,
                             levels=levels, transform=ccrs.PlateCarree(),
                             cmap=cmap, extend="both", zorder=1)
        else:
            cs = ax.contourf(lon, lat, data3,
                             levels=levels, transform=ccrs.PlateCarree(),
                             cmap=cmap, extend="both", zorder=1)
            data4 = np.ma.array(data4, mask=lcc != 130)
            cs2 = ax.contourf(lon, lat, data4,
                              levels=[0, 0.05, 1], transform=ccrs.PlateCarree(),
                              hatches=['..', None], colors="none", zorder=2)
            
        if i==2 or i==3:
            ax.set_yticklabels([])

    cbar_ax = fig.add_axes([0.3, 0.06, 0.4, 0.04])
    cb = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
    cb.ax.tick_params(labelsize=15)

    # plt.suptitle(f'{title}', fontsize=20)
    # plt.savefig(
    #     rf'E:/SPEI_base/python MG/JPG/{title}.jpg', bbox_inches='tight')
    # plt.savefig(
    #     rf'F:/000000 论文图表最终版/{title}.jpg', bbox_inches='tight')
    plt.show()


# %%
def PLOT(data1, data2, data3, data4):

    print('5 percentile is: ', np.nanpercentile(data3, 5))
    print('95 percentile is: ', np.nanpercentile(data3, 95))

    levels = np.arange(-1, 1.01, 0.1)
   
    #########
    cmap = cmaps.BlueWhiteOrangeRed_r
    cmap = plt.get_cmap(cmap)
    newcolors = cmap(np.linspace(0, 1, 254))
    red = newcolors[:107]
    blue = newcolors[148:]
    new = np.vstack((red, blue))
    newcmap = ListedColormap(new)
    plt.cm.register_cmap(name='mycmp', cmap=newcmap)
    
    title = r"MG SPEI03 GSL Diff (81-99)(00-20)"

    Creat_figure(data1, data2, data3, data4, levels, newcmap, lat, lon, title)


# %%


read_nc()
spei = read_nc2()

spei1 = spei[1:19, :, :]  # 81-99
spei2 = spei[19:, :, :]  # 00-18

spei1_ave = np.nanmean(spei1, axis=0)
spei2_ave = np.nanmean(spei2, axis=0)
spei_diff = spei2_ave-spei1_ave

_, pt = tTest(spei1, spei2)
PLOT(spei1_ave, spei2_ave, spei_diff, pt)
