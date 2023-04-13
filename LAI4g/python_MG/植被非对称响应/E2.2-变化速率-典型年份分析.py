# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 15:20:28 2023

可能部分年份植被变化较小，合成的话效果不是很明显
应将典型年份合成

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
from matplotlib.patches import Rectangle

import cmaps
import warnings

import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['times new roman']  # 指定默认字体

warnings.filterwarnings("ignore")


# %%
# yr = np.arange(1983, 2021)

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
        lai_diff = lai_gs[1:, ]-lai_gs[:-1, ]
    return lai_diff


def read_nc2():
    inpath = (r"E:/SPEI_base/data/spei03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        spei = (f.variables['spei'][972:, :, :])
        spei = spei.reshape(39, 12, 30, 50)
        spei_gs = np.nanmean(spei[:, 4:9, :, :], axis=1)
        spei_diff = spei_gs[1:, ]-spei_gs[:-1, ]
    return spei_diff


# %%
lcc = read_lcc()
lai = read_nc()
spei = read_nc2()
cr = lai/spei

lai1, lai2 = lai[:18, ], lai[18:, ]
spei1, spei2 = spei[:18, ], spei[18:, ]
cr1, cr2 = cr[:18, ], cr[18:, ]

# %%


def drop_nongrass(data):
    mask = lcc != 130
    sp = data.shape[0]
    data_mk = np.empty((sp, 30, 50))

    for i in range(sp):
        a = np.ma.array(data[i], mask=mask, fill_value=-999)
        data_mk[i] = a.filled()

    return data_mk


def drop_nongrass2(data):
    mask = lcc != 130
    data_mk = np.empty((30, 50))

    a = np.ma.array(data, mask=mask, fill_value=-999)
    data_mk = a.filled()

    return data_mk


# %%
def corr(d1, d2):
    sp = d1.shape
    spi, spj = sp[1], sp[2]
    r, p = np.zeros((spi, spj)), np.zeros((spi, spj))
    for i in range(spi):
        for j in range(spj):
            a = d1[:, i, j]
            b = d2[:, i, j]
            if lcc[i, j] != 130:
                r[i, j], p[i, j] = -999, -999
            else:
                if np.isnan(a).any() or np.isnan(b).any():
                    r[i, j], p[i, j] = -787, -787
                else:
                    r[i, j], p[i, j] = pearsonr(a, b)
    return r, p


def drop_nondrought_dominat(data1, data2, data):
    global r, p
    r, p = corr(data1, data2)
    # sp = data.shape[0]
    # data_mk = np.empty((sp, 30, 50))

    # for i in range(sp):
    #     a = np.ma.array(data[i], mask=mask, fill_value=-999)
    #     data_mk[i] = a.filled()
    data[((r > -1) & (r < 0)) | (p > 0.05)] = 100
    return data


# %%


def typical_synthesis(spei, lai, cr):
    cr_wet = cr.copy()
    cr_wet[~(spei > 0.8)] = np.nan
    wet = np.nanmean(cr_wet, axis=0)
    wet_mk = drop_nongrass2(wet)
    wet_mk2 = drop_nondrought_dominat(spei, lai, wet_mk)

    cr_dry = cr.copy()
    cr_dry[~(spei < -0.8)] = np.nan
    dry = np.nanmean(cr_dry, axis=0)
    dry_mk = drop_nongrass2(dry)
    dry_mk2 = drop_nondrought_dominat(spei, lai, dry_mk)

    return wet_mk2, dry_mk2


def non_typical_synthesis(spei, lai, cr):
    cr_wet = cr.copy()
    cr_wet[~((spei > 0) & (spei < 0.8))] = np.nan
    wet = np.nanmean(cr_wet, axis=0)
    wet_mk = drop_nongrass2(wet)
    wet_mk2 = drop_nondrought_dominat(spei, lai, wet_mk)

    cr_dry = cr.copy()
    cr_dry[~((spei > -0.8) & (spei < 0))] = np.nan
    dry = np.nanmean(cr_dry, axis=0)
    dry_mk = drop_nongrass2(dry)
    dry_mk2 = drop_nondrought_dominat(spei, lai, dry_mk)

    return wet_mk2, dry_mk2

# %%


def cut_counts(data):
    data[np.isnan(data)] = 0
    ind = lcc == 130
    data = data[ind]

    bin = [data.min(), -0.01, -0.0001, 0.0001, 0.01, data.max()]
    labels = ["Dry asymmetry", "Symmetric",
              "Non-drought dominant", "Symmetric2", "Wet asymmetry"]
    cata = pd.cut(data, bin, right=False, labels=labels)

    count = cata.value_counts()/383
    count[1] = count[1]+count[-2]
    del count["Symmetric2"]

    count2 = pd.DataFrame(dict(Percentage=count, sort=list("3214")))
    count2.sort_values(by=["sort"], inplace=True)
    print(count2)
    print()

    return count2

# %%


def percentile(data):
    per = np.arange(5, 95.1, 5)
    for p in per:
        _ = np.nanpercentile(data.data, p)
        print(f"{p}分位数为：", _)
    print("\n")


# %% 显著变化年份
cr2_wet, cr2_dry = typical_synthesis(spei2, lai2, cr2)
cr2_diff = cr2_wet-cr2_dry
print("2001-2020", "Significant change")
cr2_count = cut_counts(cr2_diff)


cr1_wet, cr1_dry = typical_synthesis(spei1, lai1, cr1)
cr1_diff = cr1_wet-cr1_dry
print("1983-2000", "Significant change")
cr1_count = cut_counts(cr1_diff)


# %%
def sub_ax(fig, pos, df):
    ax = fig.add_axes(pos)
    color = ["lightgray", "k", "red", "dodgerblue"]
    color2 = ["#1681FC", "#FD7F1E"]

    df.plot.bar(ax=ax, rot=0, linewidth=2,
                edgecolor=None, color=color2, alpha=0.5)
    df.plot.bar(ax=ax, rot=0, linewidth=2,
                edgecolor=color, facecolor=(0, 0, 0, 0))

    ax.set_ylim(0, 0.6)
    yticks = np.arange(0, 0.61, 0.15)
    ax.set_yticks(yticks)
    ylabels = [f"{100*x:.0f}%" for x in yticks]
    ax.set_yticklabels(ylabels)
    ax.set_ylabel("Area Percentage", fontsize=10)
    ax.set_title(f"Comparison", fontsize=12, loc="left")

    return ax


def bar_plot(df1, df2, title):
    fig = plt.figure(figsize=(8, 6), dpi=1080)
    pos = [0.1, 0.1, 0.6, 0.6]

    ax2 = fig.add_axes([0.25, 0.58, 0.2, 0.1])

    df1["sort"] = df2["Percentage"]
    ax1 = sub_ax(fig, pos, df1)

    # 图例
    global children
    children = plt.gca().get_children()
    handles = children[0:5:4]+children[8:12]
    labels = ['Before 1999', 'After 1999']
    [labels.append(x) for x in df1.index]
    # ax1.legend(handles, labels, fontsize=10, loc='best', ncol=2)
    ax2.legend(children[0:5:4], ['Before 1999', 'After 1999'],
               fontsize=10, loc='upper right', ncol=2, bbox_to_anchor=(1, 1.3, 0.4, 0.5))
    ax1.legend(children[8:12], df1.index, fontsize=10, loc='best')
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles, ['Before 1999', 'After 1999'], fontsize=8)

    plt.savefig(rf'E:/LAI4g/JPG_MG/{title}.jpg',
                bbox_inches='tight')

bar_plot(cr1_count, cr2_count, title=r"Change Ratio Significant Comparison")

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

    ax.set_yticks(list(range(42, 55, 3)))
    ax.set_xticks(list(np.arange(102, 126, 5)))  # 需要显示的纬度

    ax.tick_params(labelsize=15)

    # 区域

    region = [100.25, 124.75, 40.25, 54.75]
    ax.set_extent(region, crs=ccrs.PlateCarree())

    # 设置网格点
    ylocs = list(range(42, 55, 3))
    xlocs = list(np.arange(102, 126, 5))
    lb = ax.gridlines(draw_labels=True, ylocs=ylocs, xlocs=xlocs, x_inline=False, y_inline=False,
                      linewidth=1, color='lightgray', linestyle='--', alpha=0.8)  # alpha是透明度

    lb.top_labels = False
    lb.bottom_labels = False
    lb.right_labels = False
    lb.left_labels = False
    lb.xlabel_style = {'size': 15}  # 修改经纬度字体大小
    lb.ylabel_style = {'size': 15}

    return ax


def Creat_figure(data1, levels, cmap, lat, lon, title):
    fig = plt.figure(figsize=(8, 6), dpi=500)
    proj = ccrs.PlateCarree()

    fig.subplots_adjust(left=0.1, bottom=0.2, right=0.9,
                        top=0.9, wspace=0.1, hspace=0.05)

    lon, lat = np.meshgrid(lon, lat)

    for i in range(1, 3):
        ax = fig.add_subplot(1, 2, i, projection=proj)
        ax = make_map(ax)

        norm = mpl.colors.BoundaryNorm(levels, cmap.N)  # 生成索引
        data1[i-1][lcc!=130]=np.nan
        cs = ax.pcolormesh(lon, lat, data1[i-1], norm=norm, cmap=cmap,
                           transform=ccrs.PlateCarree(), zorder=1)

        # lcc2 = np.ma.array(lcc, mask=lcc != 200)
        # ax.contourf(lon, lat, lcc2,
        #             transform=ccrs.PlateCarree(),
        #             colors="lightgray", zorder=2, corner_mask=False)

        ax.tick_params(labelsize=15)
        if i == 1:
            ax.set_title(f"(a) Before 1999",
                         loc="left", fontsize=15)
        elif i == 2:
            ax.set_title(f"(b) After 1999", loc="left", fontsize=15)
            ax.set_yticklabels([])

    # cbar_ax = fig.add_axes([0.2, 0.1, 0.6, 0.02])
    # cb = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
    # cb.ax.tick_params(labelsize=15)
    
    color_list = ["lightgray", "k",'r',"dodgerblue"]
    labels = ['Non-drought dominant', 'Symmetric', 'Dry asymmetry', 'Wet asymmetry']
    # 绘制矩形图例
    rectangles = [Rectangle((0, 0,), 1, 1, facecolor=x, edgecolor="w") for x in color_list]
    labels = labels
    ax.legend(rectangles, labels,
              bbox_to_anchor=(1.65, 0.5), fancybox=True, frameon=True)
    

    # plt.suptitle(f'{title}', fontsize=25)
    plt.savefig(
        rf'E:/LAI4g/JPG_MG/{title}.jpg', bbox_inches='tight')
    plt.show()


def read_plot(data):
    levels = [-10, -0.01, -0.0001, 0.0001, 0.01, 10]
    
    color = ["red", "k", "lightgray", "k", "dodgerblue"]
    cmap = ListedColormap(color)
    # cmap = cmaps.CBR_set3
    Creat_figure(data, levels, cmap, lat, lon,
                 title=f"Change Ratio Significant Comparison Pcolor")


cr2_plot = [cr1_diff, cr2_diff]
read_plot(cr2_plot)


# %% 非显著变化年份
# cr2_wet_non, cr2_dry_non = non_typical_synthesis(spei2, lai2, cr2)
# cr2_diff_non = cr2_wet_non-cr2_dry_non
# print("2001-2020", "Non-Significant change")
# percentile(cr2_diff_non)
# cr2_count_non = cut_counts(cr2_diff_non)


# cr1_wet_non, cr1_dry_non = non_typical_synthesis(spei1, lai1, cr1)
# cr1_diff_non = cr1_wet_non-cr1_dry_non
# print("1983-2000", "Non-Significant change")
# percentile(cr1_diff_non)
# cr1_count_non = cut_counts(cr1_diff_non)


# %% 简易画图测试
if __name__ == "__main__":
    def percentile(data):
        per = np.arange(5, 95.1, 5)
        for p in per:
            _ = np.nanpercentile(data.data, p)
            print(f"{p}分位数为：", _)


    def plot(data, level, cmap, title):
        plt.figure(1, dpi=500)
        print(title)
        data[~(lcc == 130)] = np.nan
        percentile(data)
        # level = [0, 0.05, np.nanmax(p2)]
        # level = [0, 0.05]
        norm = mpl.colors.BoundaryNorm(level, cmap.N)
        cs = plt.pcolor(data, cmap=cmap, norm=norm)
        plt.colorbar(cs, shrink=0.75, orientation='horizontal')
        plt.title(f"{title}")
        plt.show()


    lev = np.arange(-0.1, 0.11, 0.01)
    plot(cr2_diff, lev, cmaps.MPL_bwr_r, "Dry-Wet")


# %% symmetric/asymmetry 阈值
# if __name__=="__main__":
#     thre = np.nanstd(cr, axis=0)
