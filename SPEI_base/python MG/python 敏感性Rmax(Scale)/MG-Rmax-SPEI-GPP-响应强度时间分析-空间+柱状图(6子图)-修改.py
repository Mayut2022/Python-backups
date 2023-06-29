# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 11:14:13 2022
abcdef
a\b是Rmax平均，c是区域平均变化；
d/e是14年中所有RmaxScale所占的比例，f是差值


@author: MaYutong
"""


from collections import Counter
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
import matplotlib.dates as mdate
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
import xarray as xr
import xlsxwriter

import seaborn as sns

from scipy.stats import linregress
from scipy.stats import pearsonr

# %%


def read_nc(inpath):
    global lat, lon
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())

        r = (f.variables['Rmax'][:])
        r_s = (f.variables['Rmax_scale'][:])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])

        r_c = r.copy()
        r_c[r_c >= 0] = 1
        r_c[r_c < 0] = -1

    return r, r_s, r_c

# %%


def read_nc3():
    global lcc
    # global a1, a2, o1, o2
    inpath = (r"E:/ESA/ESA_LUC_2000_SPEI03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        # print(f.variables['lccs_class'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        lcc = (f.variables['lcc'][:])

# %%


def lcc_count(data):
    sp = data.shape
    data = data.reshape(1, sp[0]*sp[1]*sp[2])
    data = np.squeeze(data)

    lcc_c = Counter(data)
    lcc_c = pd.Series(lcc_c)
    lcc_c = lcc_c.sort_index()

    return lcc_c.iloc[:12]


def cut(data):
    '''
    sp = data.shape
    data = data.reshape(1, sp[0]*sp[1]*sp[2])
    data = np.squeeze(data)
    '''
    bin = np.arange(-1, 1.1, 0.1)
    labels = np.arange(-0.95, 0.96, 0.1)
    cats = pd.cut(data, bin, right=False, labels=labels)

    value = cats.value_counts()
    sort = value.sort_values(ascending=False)
    return sort.index[0]

# %% mask数组, mask掉除130以外的


def mask(x, data):
    lcc2 = np.array(lcc)

    np.place(lcc2, lcc2 != x, 1)
    np.place(lcc2, lcc2 == x, 0)

    spei_ma = ma.masked_array(data, mask=lcc2)

    return spei_ma


def mask2(x, data):
    lcc2 = np.array(lcc)

    np.place(lcc2, lcc2 != x, 1)
    np.place(lcc2, lcc2 == x, 0)

    spei_ma = np.empty((28, 30, 50))
    spei_ma = np.ma.array(spei_ma)

    for i in range(28):
        a = data[i, :, :]
        a = ma.masked_array(a, mask=lcc2, fill_value=np.nan)
        spei_ma[i, :, :] = a.filled()

    return spei_ma

# %%


def make_map(ax):
    # 打开边框
    ax.spines['geo'].set_visible(True)

    # 设置cfeature自带底图
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1)
    ax.add_feature(cfeature.RIVERS.with_scale('50m'), linewidth=1, zorder=2)

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
    ax.add_feature(provinces, linewidth=1.2, zorder=3)
    ax.add_feature(world, linewidth=1.2, zorder=3)

    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    ax.set_yticks(list(range(42, 55, 3)))
    ax.set_xticks(list(np.arange(102.5, 126, 5)))  # 需要显示的纬度

    ax.tick_params(labelsize=15)

    # 区域

    region = [100.25, 124.75, 40.25, 54.75]
    ax.set_extent(region, crs=ccrs.PlateCarree())

    # 设置网格点
    ylocs = list(range(42, 55, 3))
    xlocs = list(np.arange(102.5, 126, 5))
    lb = ax.gridlines(draw_labels=True, ylocs=ylocs, xlocs=xlocs, x_inline=False, y_inline=False,
                      linewidth=1, color='lightgray', linestyle='--', alpha=0.8)  # alpha是透明度

    lb.top_labels = False
    lb.bottom_labels = False
    lb.right_labels = False
    lb.left_labels = False
    lb.xlabel_style = {'size': 15}  # 修改经纬度字体大小
    lb.ylabel_style = {'size': 15}

    return ax

####################################################


def Creat_figure(data1, data2, data4, data5, levels, cmap, lat, lon, title):
    fig = plt.figure(figsize=(18, 8), dpi=500)
    proj = ccrs.PlateCarree()

    fig.subplots_adjust(left=0.1, bottom=0.15, right=0.9,
                        top=0.9, wspace=0.2, hspace=0.2)

    lon, lat = np.meshgrid(lon, lat)

    data3 = np.nanmean(r_mask, axis=(1, 2))
    data6 = data5-data4

    yr_str1 = ["82-99 ave", "00-18 ave", ""]
    yr_str = ["82-99", "00-18", "DIFF"]
    for i in range(1, 4):
        if i != 3:
            ax = fig.add_subplot(2, 3, i, projection=proj)
            ax = make_map(ax)
            ax.set_title(f"Rmax {yr_str1[i-1]}",
                         fontsize=15, loc="left", color="b")
            cs = eval(
                f'ax.contourf(lon, lat, data{i}, levels=levels, transform=ccrs.PlateCarree(), cmap=cmap, extend="both", zorder=1)')
            lcc2 = np.ma.array(lcc, mask=lcc!=200)
            cs2 = ax.contourf(lon, lat, lcc2, transform=ccrs.PlateCarree(), colors="lightgray", zorder=1, corner_mask=False)
        else:
            ax = fig.add_subplot(2, 3, i)
            # t = pd.date_range(f'1986', periods=28, freq='1AS-JAN')
            t2 = np.arange(1, 29)
            ###################
            ax2 = ax.twinx()
            ax2.scatter(t2, data3, zorder=2, s=10, c="k")
            ax2.plot(t2, data3, zorder=2, c="k", label="Rmax ave")
            ax2.tick_params(labelsize=15)
            ax2.set_ylim(0.4, 0.75)
            ax2.set_yticks(np.arange(0.4, 0.76, 0.05))
            ################
            data3_anom = data3-data3.mean()
            # 渐变色柱状图
            # 归一化
            norm = plt.Normalize(-0.15, 0.15)  # 值的范围
            norm_values = norm(data3_anom)
            map_vir = cm.get_cmap(name='RdBu')
            colors = map_vir(norm_values)

            ax.axhline(0, color='k', linewidth=0.8)
            ax.bar(t2, data3_anom, color=colors, width=1,
                   edgecolor="w", label="Rmax anom")
            ax.set_ylim(-0.16, 0.16)
            ax.set_yticks(np.arange(-0.16, 0.17, 0.04))
            ax.set_xticks(t2[2::4])
            ax.set_xticklabels(np.arange(1988, 2013, 4))
            '''
            tt = pd.date_range(f'1986', periods=7, freq='4AS-JAN')
            ax.set_xticks(tt)
            ax.xaxis.set_major_formatter(mdate.DateFormatter('%y'))
            '''
            ax.set_title(f"Moving Rmax ave", fontsize=15,
                         loc="left", color="b")
            line1, label1 = ax.get_legend_handles_labels()
            line2, label2 = ax2.get_legend_handles_labels()
            lines = line1+line2
            labels = label1+label2
            ax.legend(lines, labels, loc='upper left', fontsize=12)
        ax.tick_params(labelsize=15)

    scale = [str(x).zfill(2) for x in range(1, 13)]
    for j in range(4, 7):
        ax = fig.add_subplot(2, 3, j)
        if j == 6:
            data6 = data6/data4.sum()
            # 渐变色柱状图
            # 归一化
            norm = plt.Normalize(-0.15, 0.15)  # 值的范围
            norm_values = norm(data6)
            map_vir = cm.get_cmap(name='BrBG')
            colors = map_vir(norm_values)
            ax.axhline(0, color='k', linewidth=0.8)
            ax.bar(scale, data6, color=colors, width=0.8, edgecolor="k")
            ax.set_ylim(-0.12, 0.12)
            ax.set_yticks(np.arange(-0.12, 0.13, 0.04))
        else:
            data = eval(f"data{j}/data{j}.sum()")
            ax.bar(scale, data, color='orange', alpha=0.5,
                   width=0.8, edgecolor='orange')
            ax.set_ylim(0, 0.4)
            yticks = np.arange(0, 0.41, 0.1)
            ax.set_yticks(yticks)
            ylabels = [f"{100*x:.0f}%" for x in yticks]
            ax.set_yticklabels(ylabels)

        ax.set_title(f"RmaxScale {yr_str[j-4]}",
                     fontsize=15, loc="left", color="b")
        ax.tick_params(labelsize=15)
        # ax.grid(which="major", color="dimgray", linestyle='--', linewidth=0.3)
        if j == 4:
            ax.set_ylabel("Area Percentage", fontsize=20)
        elif j == 5:
            ax.set_xlabel("SPEI Time Scale", fontsize=20)

    cbar_ax = fig.add_axes([0.95, 0.2, 0.015, 0.6])
    cb = fig.colorbar(cs, cax=cbar_ax, orientation='vertical')
    # cb.set_label('Units: mm/month (CRU-GLEAM)', fontsize=15)
    cb.ax.tick_params(labelsize=15)

    plt.suptitle(f'{title}', fontsize=25)
    plt.savefig(
        rf'E:/SPEI_base/python MG/JPG/Rmax_Scale分析/{title}.jpg', bbox_inches='tight')
    plt.show()


# %%
read_nc3()
df2 = pd.read_excel('E:/ERA5/每月天数.xlsx')
mn_str = df2['月份']
# %%
# r_s_cou = lcc_count(r_s)
# r_s_mask_cou = lcc_count(r_s_mask)

# %% 画图


def read_plot(data1, data2, data3, data4, title):
    print('5 percentile is: ', np.nanpercentile(data1, 5))
    print('95 percentile is: ', np.nanpercentile(data1, 95), "\n")

    print(np.nanmax(data3))
    print(np.nanmin(data3), "\n")

    levels = np.arange(0.4, 0.81, 0.05)
    cmap = cmaps.MPL_YlGn
    Creat_figure(data1, data2, data3, data4, levels, cmap, lat, lon, title)


def most_data(data):
    data_cut = np.zeros((30, 50))
    for i in range(30):
        for j in range(50):
            if np.isnan(data[:, i, j]).any():
                data_cut[i, j] = np.nan
            else:
                data_cut[i, j] = cut(data[:, i, j])

    return data_cut


# %% 修改绘图 将上面三幅空间图分别修改为相关性平均和相关性趋势
# month = np.arange(4, 11, 1)

# for mn in month[2:5]:
#     print(f"month {mn} is in programming!")
#     inpath = rf"E:/SPEI_base/python MG/response time data/GPP_SPEI_MG_month{mn}.nc"
#     r, r_s, _ = read_nc(inpath)
#     r_mask, r_s_mask = mask2(130, r), mask2(130, r_s)
#     r_1 = r_mask[:14, :, :]
#     r_2 = r_mask[14:, :, :]

#     r_1_ave, r_2_ave = r_1.mean(axis=0), r_2.mean(axis=0)

#     r_s_1 = r_s_mask[:14, :, :]
#     r_s_2 = r_s_mask[14:, :, :]

#     cou1 = lcc_count(r_s_1)
#     cou2 = lcc_count(r_s_2)

#     read_plot(r_1_ave, r_2_ave, cou1, cou2,
#               title=fr"3 GPP Rmax Scale {mn_str[mn-1]}")

# %% Summer
# inpath = rf"E:/SPEI_base/python MG/response time data/GPP_SPEI_MG_Summer.nc"
# r, r_s, _ = read_nc(inpath)
# r_mask, r_s_mask = mask2(130, r), mask2(130, r_s)
# r_1 = r_mask[:14, :, :]
# r_2 = r_mask[14:, :, :]

# r_1_ave, r_2_ave = r_1.mean(axis=0), r_2.mean(axis=0)

# r_s_1 = r_s_mask[:14, :, :]
# r_s_2 = r_s_mask[14:, :, :]

# cou1 = lcc_count(r_s_1)
# cou2 = lcc_count(r_s_2)


# read_plot(r_1_ave, r_2_ave, cou1, cou2,
#             title=fr"GPP Rmax Scale Summer")

# %% 修改图，只将非植被区mask掉
inpath = rf"E:/SPEI_base/python MG/response time data/GPP_SPEI_MG_Summer.nc"
r, r_s, _ = read_nc(inpath)
r_mask, r_s_mask = mask2(130, r), mask2(130, r_s)
r_1 = r_mask[:14, :, :]
r_2 = r_mask[14:, :, :]

r_1_ave, r_2_ave = r_1.mean(axis=0), r_2.mean(axis=0)

r_s_1 = r_s_mask[:14, :, :]
r_s_2 = r_s_mask[14:, :, :]

cou1 = lcc_count(r_s_1)
cou2 = lcc_count(r_s_2)

# mask部分
r_1_plot = r[:14, :, :]
r_2_plot = r[14:, :, :]

r_1_plot, r_2_plot = r_1_plot.mean(axis=0), r_2_plot.mean(axis=0)

read_plot(r_1_plot, r_2_plot, cou1, cou2,
          title=fr"GPP Rmax Scale Summer (Mask)")
