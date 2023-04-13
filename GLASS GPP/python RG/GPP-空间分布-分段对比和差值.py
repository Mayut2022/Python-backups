# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 12:01:10 2023

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
    # global a1, a2, o1, o2
    inpath = (r"E:/ESA/ESA_LUC_2000_SPEI03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        # print(f.variables['lccs_class'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        lcc = (f.variables['lcc'][:])

# %%


def sif_xarray(band1):
    mn = np.arange(12)
    sif = xr.DataArray(band1, dims=['mn', 'y', 'x'], coords=[mn, lat2, lon2])

    sif_MG = sif.loc[:, 40:55, 100:125]
    sif_MG_gsl = np.nansum(sif_MG[4:9, :, :], axis=0)
    # sif_MG_su = np.nanmean(sif_MG[6:9, :, :], axis=0)
    return np.array(sif_MG_gsl)


# %%


def read_nc2(inpath):
    global lat2, lon2
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())

        gpp = (f.variables['GPP'][:])
        lat2 = (f.variables['lat'][:])
        lon2 = (f.variables['lon'][:])

    return gpp


def exact_data1():
    for yr in range(1982, 2019):
        inpath = rf"E:/GLASS-GPP/Month RG/GLASS_GPP_RG_SPEI_0.5X0.5_{yr}.nc"
        data = read_nc2(inpath)

        data_MG = sif_xarray(data)
        data_MG = data_MG.reshape(1, 30, 50)
        if yr == 1982:
            data_all = data_MG
        else:
            data_all = np.vstack((data_all, data_MG))

    return data_all


# %%
def read_nc3():
    inpath = (r"E:/GLASS-GPP/MG detrend Zscore/MG_GPP_Zscore_SPEI_0.5x0.5.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        # print(f.variables['GPP_Z'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        gpp = (f.variables['GPP_Z'][:, 4:9, :])
        gpp1, gpp2 = gpp[:18, ], gpp[18:, ]

        gpp11 = np.nanmean(gpp1, axis=(1))
        gpp22 = np.nanmean(gpp2, axis=(1))

    return gpp11, gpp22
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


def Creat_figure(data1, data2, data3, data4, levels, cmap, lat, lon, title):
    fig = plt.figure(figsize=(16, 4), dpi=1080)
    proj = ccrs.PlateCarree()

    fig.subplots_adjust(left=0.12, bottom=0.12, right=0.88,
                        top=0.9, wspace=0.22, hspace=0.27)

    lon, lat = np.meshgrid(lon, lat)
    title_str = ["(a) 1982-1999 Ave", "(b) 2000-2018 Ave", "(c) DIFF(b-a)"]
    for i in range(1, 4):
        ax = fig.add_subplot(1, 3, i, projection=proj)
        ax = make_map(ax)
        ax.set_title(f"{title_str[i-1]}", fontsize=15, loc="left")
        lcc2 = np.ma.array(lcc, mask=lcc != 200)
        ax.contourf(lon, lat, lcc2,
                    transform=ccrs.PlateCarree(),
                    colors="lightgray", zorder=2, corner_mask=False)
        if i == 1:
            cs = ax.contourf(lon, lat, data1,
                             levels=levels, transform=ccrs.PlateCarree(),
                             cmap=cmap, extend="both", zorder=1, corner_mask=False)
        elif i == 2:
            cs = ax.contourf(lon, lat, data2,
                             levels=levels, transform=ccrs.PlateCarree(),
                             cmap=cmap, extend="both", zorder=1, corner_mask=False)
        else:
            levels2 = np.arange(-0.8, 0.81, 0.08)
            cmap2 = cmaps.MPL_BrBG[20:109]
            cs2 = ax.contourf(lon, lat, data3,
                              levels=levels2, transform=ccrs.PlateCarree(),
                              cmap=cmap2, extend="both", zorder=1, corner_mask=False)
            data4 = np.ma.array(data4, mask=lcc == 200)
            cs3 = ax.contourf(lon, lat, data4,
                              levels=[0, 0.05, 1], transform=ccrs.PlateCarree(),
                              hatches=['..', None], colors="none", zorder=2)

    ######
    cbar_ax = fig.add_axes([0.23, 0.06, 0.3, 0.04])
    cb = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
    cb.ax.tick_params(labelsize=15)
    # cb.set_label('Units: g C/m2/GSL', fontsize=15)
    ######
    cbar_ax = fig.add_axes([0.675, 0.06, 0.2, 0.04])
    cb = fig.colorbar(cs2, cax=cbar_ax, orientation='horizontal')
    cb.ax.tick_params(labelsize=15)
    # cb.set_label('Units: g C/m2/GSL', fontsize=15)

    plt.suptitle(f'{title}', fontsize=20)
    plt.savefig(
        rf'E:/GLASS-GPP/JPG MG/{title}.jpg', bbox_inches='tight')
    plt.show()


# %%
def PLOT(data1, data2, data3, data4):

    print('5 percentile is: ', np.nanpercentile(data1, 5))
    print('95 percentile is: ', np.nanpercentile(data1, 95))

    levels = np.arange(0, 1001, 50)
    cmap = "YlGn"

    # title = r"MG GPP Summer Diff (81-99)(00-18)"
    title = r"MG GPP GSL Diff (81-99)(00-18)"

    Creat_figure(data1, data2, data3, data4, levels, cmap, lat, lon, title)
    
    
#%%
def PLOT2(data1, data2, data3, data4):

    print('5 percentile is: ', np.nanpercentile(data2, 5))
    print('95 percentile is: ', np.nanpercentile(data2, 95), "\n")

    levels = np.arange(-0.5, 0.51, 0.05)
    cmap = "YlGn"
    
    print('5 percentile is: ', np.nanpercentile(data3, 5))
    print('95 percentile is: ', np.nanpercentile(data3, 95))

    # title = r"MG GPP Summer Diff (81-99)(00-18)"
    title = r"MG GPPZ GSL Ave Diff (81-99)(00-18)"

    Creat_figure(data1, data2, data3, data4, levels, cmap, lat, lon, title)


# %% Summer
# read_nc()
# gpp = exact_data1()

# gpp1 = gpp[:18, :, :]  # 81-99
# gpp2 = gpp[19:, :, :]  # 00-18

# gpp1_ave = np.nanmean(gpp1, axis=0)
# gpp2_ave = np.nanmean(gpp2, axis=0)
# gpp_diff = gpp2_ave-gpp1_ave

# _, pt = tTest(gpp1, gpp2)
# PLOT(gpp1_ave, gpp2_ave, gpp_diff, pt)

# %% 生长季总和 gpp原值
read_nc()
# gpp = exact_data1()

# gpp1 = gpp[:18, :, :]  # 81-99
# gpp2 = gpp[19:, :, :]  # 00-18

# gpp1_ave = np.nanmean(gpp1, axis=0)
# gpp2_ave = np.nanmean(gpp2, axis=0)
# gpp_diff = gpp2_ave-gpp1_ave

# _, pt = tTest(gpp1, gpp2)
# PLOT(gpp1_ave, gpp2_ave, gpp_diff, pt)


#%% 生长季平均 gpp-Zscore
gppz1, gppz2 = read_nc3()
gppz1_ave = np.nanmean(gppz1, axis=0)
gppz2_ave = np.nanmean(gppz2, axis=0)
gppz_diff = gppz2_ave-gppz1_ave

_, pt = tTest(gppz1, gppz2)
PLOT2(gppz1_ave, gppz2_ave, gppz_diff, pt)