# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 15:57:08 2023

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

from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from scipy.stats import pearsonr

plt.rcParams['font.sans-serif'] = ['times new roman']  # 指定默认字体

# %% 480(输入data) -> 月 年


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

# %%


def read_nc(inpath):
    global gpp, lat2, lon2
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())

        gpp = (f.variables['GPP'][:])
        lat2 = (f.variables['lat'][:])
        lon2 = (f.variables['lon'][:])

    return gpp


def sif_xarray2(band1):
    mn = np.arange(12)
    sif = xr.DataArray(band1, dims=['mn', 'y', 'x'], coords=[mn, lat2, lon2])

    sif_MG = sif.loc[:, 40:55, 100:125]
    return np.array(sif_MG)


def exact_data3():
    for yr in range(1982, 2019):
        inpath = rf"E:/GLASS-GPP/Month RG/GLASS_GPP_RG_SPEI_0.5X0.5_{yr}.nc"
        data = read_nc(inpath)

        data_MG = sif_xarray2(data)
        data_MG = data_MG.reshape(1, 12, 30, 50)

        if yr == 1982:
            data_all = data_MG
        else:
            data_all = np.vstack((data_all, data_MG))

    return data_all


# %% mask数组, mask掉200的非植被区


def mask_non(x, data):
    lcc2 = np.array(lcc)

    np.place(lcc2, lcc2 != x, 0)
    np.place(lcc2, lcc2 == x, 1)

    spei_ma = np.empty((37, 30, 50))
    spei_ma = np.ma.array(spei_ma)

    for i in range(37):
        a = data[i, :, :]
        a = np.ma.masked_array(a, mask=lcc2)
        spei_ma[i, :, :] = a

    return spei_ma


# %%


def read_nc2():
    global lat, lon
    # global a1, a2, o1, o2
    inpath = (r"E:\ESA\ESA_LUC_2000_SPEI03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        # print(f.variables['lccs_class'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        lcc = (f.variables['lcc'][:])

    return lcc

# %%


def read_nc3():
    inpath = (rf"E:/SPEI_base/data/spei03_MG_season.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        print(f.variables['spei'])
        spei = (f.variables['spei'][1, 81:-2, :, :])

    return spei


# %%
'''EOF分解 要求（time，lat，lon）分布'''


def eof_cal(data):
    coslat = np.cos(np.deg2rad(lat))
    wgts = np.sqrt(coslat)[..., np.newaxis]

    # 计算纬度权重
    solver = Eof(data, weights=wgts)
    # 创建EOF函数
    eof = solver.eofs(neofs=3)
    # 获取前三个模态
    pc = solver.pcs(npcs=3, pcscaling=1)
    var = solver.varianceFraction()
    # 获取对应的PC序列和解释方差
    # 方差贡献
    eigen_Values = solver.eigenvalues()
    percentContrib = eigen_Values * 100./np.sum(eigen_Values)
    # North 检验
    eigen = solver.eigenvalues(neigs=3)
    north = solver.northTest(neigs=3)
    print('特征值: ', solver.eigenvalues(neigs=3))
    print('标准误差: ', solver.northTest(neigs=3))

    print("第一特征值保留！")
    test = [1]
    if eigen[0] > (eigen[1]+north[0]):
        print("第二特征值过检！")
        test.append(2)

    if eigen[0] > (eigen[1]+north[0]) and eigen[1] > (eigen[2]+north[1]):
        print("第三特征值过检！")
        test.append(3)

    print("\n")
    return eof, pc, percentContrib, test

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


def CreatMap(data1, data2, lon, lat, levels, cmap, title):
    fig = plt.figure(figsize=(10, 6), dpi=1080)
    proj = ccrs.PlateCarree()

    ax = fig.add_subplot(111, projection=proj)
    fig.subplots_adjust(left=0.1, bottom=0.18, right=0.9,
                        top=0.92, wspace=None, hspace=None)

    #ax = fig.add_axes([0.1, 0.8, 0.5, 0.3], projection=proj)

    # 打开边框
    ax.spines['geo'].set_visible(True)

    # 设置cfeature自带底图
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1)
    ax.add_feature(cfeature.RIVERS.with_scale('50m'), linewidth=1.5, zorder=3)

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
    ax.add_feature(provinces, linewidth=0.8, zorder=2)
    ax.add_feature(world, linewidth=0.8, zorder=2)

    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    
    yticks = np.arange(42.5, 55, 5)
    xticks = np.arange(102.5, 130, 5)
    
    ax.set_yticks(yticks)
    ax.set_xticks(xticks)  # 需要显示的纬度
    ax.tick_params(labelsize=15)

    # 区域
    region = [100.25, 124.75, 40.25, 54.75]
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

    # 绘制填色图等
    lon, lat = np.meshgrid(lon, lat)
    cs = ax.contourf(lon, lat, data1,
                     levels=levels, transform=ccrs.PlateCarree(),
                     cmap=cmap, extend="both", zorder=1, corner_mask=False)

    cs2 = ax.contourf(lon, lat, data2,
                      levels=[0, 0.05, 1], transform=ccrs.PlateCarree(),
                      hatches=['.', None], colors="none", zorder=3)
    lcc2 = np.ma.array(lcc, mask=lcc != 200)
    ax.contourf(lon, lat, lcc2,
                transform=ccrs.PlateCarree(),
                colors="lightgray", zorder=4, corner_mask=False)
    
    cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.03])
    cb = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
    #cb.set_label('units: mm/month', fontsize=15)
    cb.ax.tick_params(labelsize=15)
    plt.suptitle(f"{title}", fontsize=20)
    plt.savefig(rf'E:/SPEI_base/python MG/JPG/EOF/{title}.jpg',
                bbox_inches='tight')
    plt.show()

# %%


def plot(data1, data2):
    print('5 percentile is: ', np.nanpercentile(data1, 5))
    print('95 percentile is: ', np.nanpercentile(data1, 95), "\n")

    levels = np.arange(0, 0.81, 0.05)
    cmap = cmaps.MPL_YlOrRd[:105]
    CreatMap(data1, data2, lon, lat, levels, cmap,
              title=f"SPEI03 corr PC1(gpp) 82-18")


# %%
gpp_MG = exact_data3()
lcc = read_nc2()
spei = read_nc3()


gpp_su = np.nanmean(gpp_MG[:, 5:8, ], axis=1)
gpp_su_mk = mask_non(200, gpp_su)
_, pc, _, _ = eof_cal(gpp_su_mk)


r, p = corr(spei, gpp_su_mk)
plot(r, p)
