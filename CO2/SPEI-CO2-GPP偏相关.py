# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 14:41:59 2023

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
import pandas as pd
import pingouin as pg

import xarray as xr
import warnings
warnings.filterwarnings("ignore")
# %% lcc


def read_lcc():
    global lat, lon
    inpath = (r"E:/ESA/ESA_LUC_2000_SPEI03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        # print(f.variables['lccs_class'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        lcc = (f.variables['lcc'][:])

    return lcc


# %% SPEI
def read_nc():
    global spei
    inpath = (r"E:/SPEI_base/data/spei03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        spei = (f.variables['spei'][972:-24, :, :])
        spei = spei.reshape(37, 12, 30, 50)
        spei_gsl = np.nanmean(spei[:, 4:9, :, :], axis=1)

    return spei_gsl
# %% CO2


def read_nc2():

    def co2_xarray(band1):
        yr = np.arange(37)
        co2 = xr.DataArray(band1, dims=['yr', 'y', 'x'], coords=[
                           yr, lat2, lon2])

        co2_MG = co2.loc[:, 40:55, 100:125]
        return np.array(co2_MG)

    ##############
    inpath = r"E:/CO2/DENG/CO2_81_18_Global_Prolong_SPEI0.5x0.5.nc"
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        lat2 = (f.variables['lat'][:])
        lon2 = (f.variables['lon'][:])
        co2 = (f.variables['co2'][12:, ])
        co2 = co2.reshape(37, 12, 360, 720)
        co2_gs = co2[:, 4:9, :, :]
        co2_gs = np.nanmean(co2_gs, axis=1)

        co2_gs_MG = co2_xarray(co2_gs)

    return co2_gs_MG


# %% GPP

def read_nc3():
    inpath = rf"E:/GLASS-GPP/Month Global SPEI0.5x0.5/MG_GPP_82_18_SPEI_0.5X0.5.nc"
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        gpp = (f.variables['GPP'][:, 4:9, :])
        gpp_gs = np.nanmean(gpp, axis=1)

    return gpp_gs


# %% 剔除SPEI
def partial_corr(spei, co2, gpp):
    pr = np.zeros((30, 50))
    p_val = np.zeros((30, 50))
    for i in range(30):
        for j in range(50):
            if np.isnan(spei[:, i, j]).any() or np.isnan(co2[:, i, j]).any() or np.isnan(gpp[:, i, j]).any():
                pr[i, j], p_val[i, j] = np.nan, np.nan
            else:
                a = spei[:, i, j]
                b = co2[:, i, j]
                d = gpp[:, i, j]
                df = pd.DataFrame(dict(SPEI=a, CO2=b, GPP=d))
                pcorr = pg.partial_corr(
                    data=df, x='CO2', y='GPP', covar='SPEI')
                pr[i, j], p_val[i, j] = pcorr['r'], pcorr['p-val']

    return pr, p_val

# %% 剔除co2
def partial_corr2(spei, co2, gpp):
    pr = np.zeros((30, 50))
    p_val = np.zeros((30, 50))
    for i in range(30):
        for j in range(50):
            if np.isnan(spei[:, i, j]).any() or np.isnan(co2[:, i, j]).any() or np.isnan(gpp[:, i, j]).any():
                pr[i, j], p_val[i, j] = np.nan, np.nan
            else:
                a = spei[:, i, j]
                b = co2[:, i, j]
                d = gpp[:, i, j]
                df = pd.DataFrame(dict(SPEI=a, CO2=b, GPP=d))
                pcorr = pg.partial_corr(
                    data=df, x='SPEI', y='GPP', covar='CO2')
                pr[i, j], p_val[i, j] = pcorr['r'], pcorr['p-val']

    return pr, p_val


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
    plt.savefig(rf'E:/CO2/JPG/JPG-MG/{title}.jpg',
                bbox_inches='tight')
    plt.show()

# %%


def plot(data1, data2):
    print('5 percentile is: ', np.nanpercentile(data1, 5))
    print('95 percentile is: ', np.nanpercentile(data1, 95), "\n")

    levels = np.arange(-0.6, 0.61, 0.05)
    cmap = cmaps.MPL_bwr_r[15:113]
    # title = r"CO2-GPP PCORR(SPEI) GSL Mean"
    title = r"SPEI-GPP PCORR(CO2) GSL Mean"
    CreatMap(data1, data2, lon, lat, levels, cmap, title)


# %%
lcc = read_lcc()
spei = read_nc()
co2 = read_nc2()
gpp = read_nc3()
# pr, p_val = partial_corr(spei, co2, gpp)
# plot(pr, p_val)

pr2, p_val2 = partial_corr2(spei, co2, gpp)
plot(pr2, p_val2)