# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 16:25:53 2023

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


def read_lcc():
    global lat, lon
    # global a1, a2, o1, o2
    inpath = (r"E:/ESA/ESA_LUC_2000_SPEI03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        # print(f.variables['lccs_class'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        lcc = (f.variables['lcc'][:])
    return lcc

# %%


def mn_yr(data):
    tmp_mn = []
    for mn in range(12):
        tmp_ = []
        for yr in range(40):
            tmp_.append(data[mn])
            mn += 12
        tmp_mn.append(tmp_)

    tmp_mn = np.array(tmp_mn)

    return tmp_mn


def MG(data):
    t = np.arange(1, 481, 1)
    l = np.arange(1, 5, 1)
    spei_global = xr.DataArray(data, dims=['t', 'l', 'y', 'x'], coords=[
                               t, l, lat2, lon2])  # 原SPEI-base数据
    spei_rg1 = spei_global.loc[:, :, 40:55, 100:125]
    spei_rg1 = np.array(spei_rg1)
    return spei_rg1


def read_nc():

    def read_smz(inpath):
        global lat2, lon2
        with nc.Dataset(inpath, mode='r') as f:
            # print(f.variables.keys())
            sm_z = (f.variables['sm_z'][:])
            lat2 = (f.variables['lat'][:])
            lon2 = (f.variables['lon'][:])
        return sm_z

    for i in range(1, 13):
        inpath = rf"E:/GLDAS Noah/DATA_RG/Zscore_Anomaly/SM_81_20_SPEI0.5x0.5_Zscore_Anomaly_month{i}.nc"
        sm_z = read_smz(inpath)
        if i == 1:
            sm_z_all = sm_z
        else:
            sm_z_all = np.vstack((sm_z_all, sm_z))

    sm_z_all_MG = MG(sm_z_all)
    sm_z_all_MG = sm_z_all_MG.reshape(12, 40, 4, 30, 50)
    sm_gsl = sm_z_all_MG[4:9, :].reshape(5*40, 4, 30, 50, order='F')
    sm_gsl1, sm_gsl2 = sm_gsl[:95, ], sm_gsl[95:, ]
    return sm_gsl1, sm_gsl2


# %%

def read_nc2():
    global lat, lon
    inpath = (r"E:/SPEI_base/data/spei03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        spei = (f.variables['spei'][960:, :])
        spei = spei.reshape(40, 12, 30, 50)
        spei_gsl = spei[:, 4:9, :].reshape(40*5, 30, 50)
        spei1, spei2 = spei_gsl[:95, :], spei_gsl[95:, :]
    return spei1, spei2


# %%
def sm_drought(x, sm, spei):
    print(str(x))

    # 非干旱统计
    # 植被值
    smm1 = sm.copy()
    exec(f"smm1[{x}]=np.nan")
    smm_N = locals()['smm1']
    gNm = np.nanmean(smm_N, axis=0)
    # 月份统计
    smc1 = sm.copy()
    exec(f"smc1[{x}]=0")
    exec(f"smc1[~({x})]=1")
    smc_N = locals()['smc1']
    gNc = np.nansum(smc_N, axis=0)

    # 干旱统计
    # 植被值
    smm2 = sm.copy()
    exec(f"smm2[~({x})]=np.nan")
    smm_Y = locals()['smm2']
    gYm = np.nanmean(smm_Y, axis=0)
    # 月份统计
    smc2 = sm.copy()
    exec(f"smc2[~({x})]=0")
    exec(f"smc2[{x}]=1")
    smc_Y = locals()['smc2']
    gYc = np.nansum(smc_Y, axis=0)

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

    title_str = ["(a) 1981-1999 Ave", "(b) 2000-2020 Ave", "(c) DIFF(b-a)"]
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
            levels2 = np.arange(-2, 2.1, 0.2)  # 前三等级
            # levels2 = np.arange(-2.5, 2.51, 0.2) #Extreme等级
            cmap2 = cmaps.MPL_bwr_r[15:113]
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
        rf'E:/GLDAS Noah/JPG_MG/Comparison2/{title}.jpg', bbox_inches='tight')
    plt.show()

# %%


def PLOT2(data1, data2, name, i):
    data3 = data2-data1
    levels = np.arange(-2, 2.1, 0.2)  # extreme
    cmap = cmaps.MPL_bwr_r[15:113]

    title = rf"{name} Drought Layer{i+1} SMZ"
    print("\n", title)
    
    print("1981-1999 Mean")
    print('5 percentile is: ', np.nanpercentile(data1, 5))
    print('95 percentile is: ', np.nanpercentile(data1, 95), "\n")
    
    print("2000-2020 Mean")
    print('5 percentile is: ', np.nanpercentile(data2, 5))
    print('95 percentile is: ', np.nanpercentile(data2, 95), "\n")

    print("DIFF")
    print('5 percentile is: ', np.nanpercentile(data3, 5))
    print('95 percentile is: ', np.nanpercentile(data3, 95), "\n")

    Creat_figure(data1, data2, data3, levels, cmap, lat, lon, title)


# %%
lcc = read_lcc()
sm1, sm2 = read_nc()
spei1, spei2 = read_nc2()

# %%
lev_d = [
    "(spei<-0.5)&(spei>=-1)", "(spei<-1)&(spei>=-1.5)", "(spei<-1.5)&(spei>=-2)", "(spei<-2)"]
lev_n = ["Mild", "Moderate", "Severe", "Extreme"]

for lev1, name in zip(lev_d[:], lev_n[:]):
    for layer in range(4):
        _, _, gYm1, _ = sm_drought(lev1, sm1[:, layer,], spei1)
        _, _, gYm2, _ = sm_drought(lev1, sm2[:, layer,], spei2)
        PLOT2(gYm1, gYm2, name, layer)