# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 11:34:04 2023

@author: MaYutong
"""
import cmaps

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
from sklearn import preprocessing
import xarray as xr

plt.rcParams['font.sans-serif'] = ['times new roman']  # 指定默认字体
# %%


def read_nc():
    global lcc, lat, lon
    # global a1, a2, o1, o2
    inpath = (r"E:\ESA\ESA_LUC_2000_SPEI03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys()) 显示所有变量
        # print(f.variables['lccs_class']) 详细显示某种变量信息
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        lcc = (f.variables['lcc'][:])

# %%


def read_nc2():
    global spei, t
    inpath = (rf"E:/SPEI_base/data/spei03_MG_season.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())

        spei = (f.variables['spei'][1 , 80:, :, ])


# %% mask数组
def mask(x, data):
    lcc2 = np.array(lcc)

    np.place(lcc2, lcc2 != x, 1)
    np.place(lcc2, lcc2 == x, 0)

    spei_ma = np.empty((40, 30, 50))
    spei_ma = np.ma.array(spei_ma)

    
    for l in range(40):
        a = data[l, :, :]
        a = ma.masked_array(a, mask=lcc2, fill_value=-999)
        spei_ma[l, :, :] = a

    spei_ma_ave = np.nanmean(spei_ma, axis=(1, 2))

    return spei_ma_ave

# %% 计算每年遭受严重干旱的面积
####### 图例，单独改和全局set用法
def area_plot(fig):
    df = pd.read_excel("E:/SPEI_base/python MG/MG Grassland干旱统计.xlsx", sheet_name="drought severity") 
    ds_per = df.iloc[:, [1, 3, 5, 7]]
    ds_per.columns=['Mild', 'Moderate', 'Severe', 'Sum area']
    
    ax = fig.add_axes([0.55, 0.1, 0.4, 0.4])
    ax.fill_between([3.5, 20], 0, 1, facecolor='dodgerblue', alpha=0.2)
    
    color = ["yellow", "orange", "red", "gray"]
    
    ds_per.plot.bar(rot=0, ax=ax, color=color, legend=True, width=0.6, alpha=0.8)
    
    ax.set_xticklabels(np.arange(1, 21, 1))
    ax.set_ylim(0, 0.9)
    yticks = np.arange(0, 0.81, 0.2)
    ax.set_yticks(yticks)
    ylabels = [f"{100*x:.0f}%" for x in yticks]
    ax.set_yticklabels(ylabels)
    ax.set_ylabel("Area Percentage", fontsize=10)
    ax.tick_params(labelsize=10)
    ax.set_title(f"Drought Serverity", fontsize=10, loc="left")
    
    ###图例
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, ncol=2, fontsize=8)
    
    return ax

# %%
def plot(data1, title):
    fig = plt.figure(figsize=(12, 4), dpi=500)

    t = np.arange(1981, 2021)
    
    ### 滑动相关
    t2 = np.arange(1985, 2016)
    data1_roll = pd.Series(data1)
    data1_roll = data1_roll.rolling(10).mean().dropna()
    
    ### 趋势线拟合
    z1 = np.polyfit(t, data1, 1)
    p1 = np.poly1d(z1)
    data1_pred = p1(t)
    
    ax = fig.add_axes([0.2, 0.1, 0.3, 0.4])
    ax.axvline(1999, color="b", linewidth=1, zorder=3)
    ax.text(1999, 1.8, "1999(turning point)", c='b')
    ax.plot(t, data1_pred, c="b", linestyle="--", linewidth=1, zorder=3)
    # 渐变色柱状图
    # 归一化
    norm = plt.Normalize(-2, 2)  # 值的范围
    norm_values = norm(data1)
    cmap = cmaps.BlueWhiteOrangeRed_r
    map_vir = cm.get_cmap(cmap)
    colors = map_vir(norm_values)
    
    ax.bar(t, data1, color=colors, label="SPEI03", zorder=2)
    ax.axhline(0, color="k", linewidth=0.3)
    # ax.plot(t2, data1_roll, c='b', zorder=3, linewidth=1)

    ax.set_title(f"Summer SPEI03", fontsize=10, loc="left")
    ax.set_ylim(-2.2, 2.2)
    ax.set_yticks(np.arange(-2, 2.1, 1))
    ax.tick_params(labelsize=10)
    
    ax2 = area_plot(fig)

    

    # fig.suptitle(f"{title}", fontsize=20)
    # plt.savefig(rf'E:/SPEI_base/python MG/JPG/{title}.jpg',
    #             bbox_inches='tight')
    plt.show()


# %% spei 夏季、秋季分布图
read_nc()
read_nc2()

spei_ave = mask(130, spei)

print(np.nanmax(spei_ave), np.nanmin(spei_ave))

plot(spei_ave, f"MG Grassland Drought Characters")


