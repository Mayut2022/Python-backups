# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 19:29:08 2022

@author: MaYutong
"""
# %%
from collections import Counter

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.io.shapereader import Reader, natural_earth

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['times new roman'] # 指定默认字体
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.ticker as mticker
from matplotlib.patches import Rectangle

import datetime as dt
import netCDF4 as nc
from matplotlib import cm
from matplotlib import font_manager
import matplotlib.dates as mdate
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd

import xarray as xr

# %%


def read_nc():
    global spei, lat, lon
    inpath = (rf"E:/SPEI_base/data/spei03_ASA_season.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        spei = (f.variables['spei'][:, 80:, :]).data

# %%


def slidet(inputdata, step):
    inputdata = np.array(inputdata)
    n = inputdata.shape[0]
    n1 = step  # n1, n2为子序列长度，需调整
    n2 = step
    t = np.zeros(n)
    for i in range(step, n-step-1):
        x1 = inputdata[i-step: i]
        x2 = inputdata[i: i+step]
        x1_mean = np.nanmean(inputdata[i-step: i])
        x2_mean = np.nanmean(inputdata[i: i+step])
        s1 = np.nanvar(inputdata[i-step: i])
        s2 = np.nanvar(inputdata[i: i+step])
        s = np.sqrt((n1 * s1 + n2 * s2) / (n1 + n2 - 2))
        t[i] = (x2_mean - x1_mean) / (s * np.sqrt(1/n1+1/n2))
    t[:step] = np.nan
    t[n-step+1:] = np.nan

    return t

# %%


def tip(data, thre):
    yr_tip = []
    a = False
    year = np.arange(1981, 2021, 1)
    for d, yr in zip(data, year):
        if np.isnan(d) == False:
            if d > thre or d < -thre:
                yr_tip.append(yr)
                # print(yr)
                a = True
    return a, yr_tip


# %% 分箱test
'''
for r in range(18):
    for c in range(40):
        test = spei[:, r, c]
        if np.isnan(test).any():
            pass
        else:
            t_move = slidet(test, N)
            a, yr_tip = tip(t_move, tt)
            title = f"Moving t-test"
            if a==True:
                print(title, "\n")


bin = np.arange(1980, 2021, 5)
cats = pd.cut(yr_tip, bin, right=True)
num = cats.value_counts()
a, *b, c = num
'''

# %%


def CreatMap(data1, lon, lat, levels, colors, title):
    fig = plt.figure(figsize=(6, 4), dpi=1080)
    proj = ccrs.PlateCarree()
    ax = fig.add_subplot(111, projection=proj)
    fig.subplots_adjust(left=0.1, bottom=0.2, right=0.9,
                        top=0.9, wspace=None, hspace=None)

    # 打开边框
    ax.spines['geo'].set_visible(True)

    # 设置cfeature自带底图
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1)
    ax.add_feature(cfeature.LAKES.with_scale('50m'), linewidth=1, zorder=2)

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
    ax.add_feature(provinces, linewidth=1, zorder=3)
    ax.add_feature(world, linewidth=1, zorder=3)

    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    yticks = np.arange(30, 51, 10)
    xticks = np.arange(40, 141, 20)

    ax.set_yticks(yticks)
    ax.set_xticks(xticks)  # 需要显示的纬度
    ax.tick_params(labelsize=15)

    # 区域
    # region = [30.25, 149.75, 25.25, 54.75]
    region = [80, 150, 25, 55]
    ax.set_extent(region, crs=ccrs.PlateCarree())

    # 设置网格点
    ylocs = yticks
    xlocs = xticks
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
                     colors=colors, extend=None, zorder=1)

    # select 矩形关键区区域
    RE = Rectangle((100, 40), 25, 15, linewidth=1.2, linestyle='-', zorder=3,
                   edgecolor='blue', facecolor='none', transform=ccrs.PlateCarree())
    ax.add_patch(RE)

    # 绘制矩形图例
    # rectangles = [Rectangle((0, 0,), 1, 1, facecolor=x, edgecolor="k") for x in colors]
    # labels = [x+1 for x in range(5)]
    # ax.legend(rectangles, labels,
    #           bbox_to_anchor=(1.1, 0.4), fancybox=True, frameon=True,
    #           fontsize=12, ncol=2, title="Number of Turning Point")

    # cbar_ax = fig.add_axes([0.3, 0.08, 0.4, 0.03])
    # cb = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
    # cb.set_label('Significant Level: 95%', fontsize=15)

    plt.suptitle(f'{title}', fontsize=15)
    plt.savefig(
        rf'E:/SPEI_base/python Asia semi-arid/JPG Season/{title}.jpg', bbox_inches='tight')
    plt.show()


# %%

read_nc()


N = 10
tt = 2.8784

# %%


def yr_cut(N, tt, spei, sn):
    global num
    bin = np.arange(1980, 2021, 5)

    for i in range(1, 9):
        exec(f"year{i} = np.zeros((60, 240))")

    for r in range(60):
        for c in range(240):
            test = spei[:, r, c]
            if np.isnan(test).any():
                pass
            else:
                t_move = slidet(test, N)
                a, yr_tip = tip(t_move, tt)
                yr_tip_cut = pd.cut(yr_tip, bin, right=True)
                num = yr_tip_cut.value_counts()
                for i, x in enumerate(num):
                    exec(f"year{i+1}[r, c] = x")

    del a, c, i, r, x, N, tt

    levels = np.arange(0.5, 6, 1)  # cut左开右闭
    colors = ['#ffd455', '#ffac00', '#ff5b00', '#ff3600', '#ff0000']

    for i, x in enumerate(num.index[2:5]):
        title = f"Moving T-test Number {x} (n=10) {season[sn-1]} 2"
        exec(f'CreatMap(year{i+1+2}, lon, lat, levels, colors, title=title)')


# %%
season = ["Spring", "Summer", "Autumn", "Winter"]
for sn in range(2, 4):
    yr_cut(N, tt, spei[sn-1, :, :, :], sn)
    print(f"{season[sn-1]} is done!")


# %%


def legend(colors):
    fig = plt.figure(figsize=(4, 2), dpi=1080)
    ax = fig.add_subplot(111)
    # 绘制矩形图例
    rectangles = [Rectangle((0, 0,), 1, 1, facecolor=x,
                            edgecolor="k") for x in colors]
    labels = [x+1 for x in range(5)]
    ax.legend(rectangles, labels,
              bbox_to_anchor=(1, 1), fancybox=True, frameon=True,
              fontsize=12, ncol=2, title="Number of Turning Point")
    # ax.spines['left'].set_visible(False)
    plt.savefig(
        rf'E:/SPEI_base/python Asia semi-arid/JPG Season/legend.jpg', bbox_inches='tight')
    plt.show()


colors = ['#ffd455', '#ffac00', '#ff5b00', '#ff3600', '#ff0000']
# legend(colors)


# %%
def region1(data):
    sn = np.arange(4)
    yr = np.arange(40)
    spei_global = xr.DataArray(data, dims=['sn', 'yr', 'y', 'x'], coords=[
                               sn, yr, lat, lon])  # 原SPEI-base数据
    spei_MG = spei_global.loc[:, :, 40:55, 100:125]

    return np.array(spei_MG)


spei_MG = region1(spei)

# %%


def count(data):
    sp = data.shape
    data = data.reshape(1, sp[0]*sp[1])
    data = np.squeeze(data)

    c = Counter(data)
    return c


def yr_cut2(N, tt, spei, sn):
    global num
    bin = np.arange(1996, 2006, 1)
    year = []
    for i in range(1, 11):
        exec(f"year{i} = np.zeros((30, 50))")

    for r in range(30):
        for c in range(50):
            test = spei[:, r, c]
            if np.isnan(test).any():
                pass
            else:
                t_move = slidet(test, N)
                a, yr_tip = tip(t_move, tt)
                yr_tip_cut = pd.cut(yr_tip, bin, right=False)
                num = yr_tip_cut.value_counts()
                for i, x in enumerate(num):
                    exec(f"year{i+1}[r, c] = x")

    for i, x in enumerate(num):
        exec(f"cc = count(year{i+1})")
        # c = locals()["cc"]
        # print(num.index[i], c)
        exec(f"year.append(cc[1]/1500)")
    # year = np.array(year)
    del a, c, i, r, x, N, tt
    return year

# %%


def bar(data, title):
    fig = plt.figure(figsize=(6, 2), dpi=1080)
    ax = fig.add_subplot(111)
    year = np.arange(1996, 2005)
    ax.bar(year, data, color='b', alpha=0.5)
    ax.tick_params(labelsize=15)
    ax.set_xticks(year)
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.set_yticklabels(["0", "20%", "40%", "60%"])
    ax.set_ylabel("Percent", fontsize=15)

    plt.savefig(
        rf'E:/SPEI_base/python Asia semi-arid/JPG Season/{title}.jpg', bbox_inches='tight')
    plt.show()


# %%
for sn in range(2, 4):
    year_count = yr_cut2(N, tt, spei_MG[sn-1, :, :, :], sn)
    bar(year_count, f"{season[sn-1]}")
    print(f"{season[sn-1]} is done!")

# %%
