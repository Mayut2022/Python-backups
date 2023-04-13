# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 15:50:43 2023

@author: MaYutong
"""


import netCDF4 as nc

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['times new roman']  # 指定默认字体

# %%


def read_lcc():
    global lcc, lat, lon
    # global a1, a2, o1, o2
    inpath = (r"E:\ESA\ESA_LUC_2000_SPEI03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys()) 显示所有变量
        # print(f.variables['lccs_class']) 详细显示某种变量信息
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        lcc = (f.variables['lcc'][:])


read_lcc()
# n = np.sum(lcc == 130) ## 统计lcc中为草地的格点数

# %%


def read_nc():
    global spei, t
    inpath = (rf"E:/SPEI_base/data/spei03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())

        spei = (f.variables['spei'][960:])
        time = (f.variables['time'][960:])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        t = nc.num2date(time, 'days since 1900-01-01 00:00:0.0').data


# %% mask数组


def mask(x, data):
    lcc2 = np.array(lcc)

    np.place(lcc2, lcc2 != x, 1)
    np.place(lcc2, lcc2 == x, 0)

    spei_ma = np.empty((480, 30, 50))
    spei_ma = np.ma.array(spei_ma)

    for l in range(480):
        a = data[l, :, :]
        a = np.ma.masked_array(a, mask=lcc2, fill_value=-9999)
        spei_ma[l, :, :] = a.filled()

    return spei_ma



# %%


def drought_cum(data):
    
    data = data.reshape(12, 1500)
    data[data == -9999] = np.nan

    co = []
    [co.append(f"ind{x+1}") for x in range(1500)]
    data_df = pd.DataFrame(data, columns=co)
    data_df = data_df.dropna(axis=1)

    # 这样计算和面积算出来相同，须进一步计算持续时间
    data_dd = data_df[np.logical_and(
        data_df < -0.5, data_df >= -1)].dropna(axis=1, how='all').sum()
    dd1 = data_dd.mean()
    data_dd2 = data_df[np.logical_and(
        data_df < -1, data_df >= -1.5)].dropna(axis=1, how='all').sum()
    dd2 = data_dd2.mean()
    data_dd3 = data_df[np.logical_and(
        data_df < -1.5, data_df >= -2)].dropna(axis=1, how='all').sum()
    dd3 = data_dd3.mean()
    data_dd4 = data_df[data_df < -2].dropna(axis=1, how='all').sum()
    dd4 = data_dd4.mean()

    dd_yr = [dd1, dd2, dd3, dd4]
    labels = ["Extreme", "Severe", "Moderate", "Mild"]
    dd_yr = pd.Series(dd_yr, index=labels[::-1])

    # 计算累计干旱强度
    # ds = data_df[np.logical_and(data_df<-0.5, data_df>=-1)]
    # ds2 = ds.dropna(axis=1, how='all')
    # ds3 = ds2.sum()
    # ds4 = ds3.mean()
    return dd_yr


# %%
read_nc()
spei_ma = mask(130, spei)
spei_mn = spei_ma.reshape(40, 12, 30, 50)

df_dd = pd.DataFrame()
for yr in range(40):
    dd_yr = drought_cum(spei_mn[yr, :, ])
    df_dd = pd.concat([df_dd, dd_yr], axis=1)

df_dd = df_dd.T
df_dd.index = np.arange(1981, 2021)
# %%


def comparison(data):
    global data1, data2
    data1 = data.iloc[:19, :]
    data2 = data.iloc[19:, :]

    data1 = data1.mean()
    data2 = data2.mean()

    data_all = pd.concat([data1, data2], axis=1)
    columns = ["Before", "After"]
    data_all.columns = columns
    return data_all


dd_cp = comparison(df_dd)

# %%


def area_plot(data, data2, title):
    global children, handles, labels
    fig = plt.figure(figsize=(12, 4), dpi=1080)

    ax = fig.add_axes([0.2, 0.1, 0.4, 0.4])

    color = ["#ffff68", "orange", "red", "#c40000"]

    ax.axvline(18, color="b", linewidth=1, zorder=2)
    ax.text(18, -16.5, "1999(turning point)", c='b')
    # linewidth控制间隙
    data.plot.bar(
        rot=0, ax=ax, color=color[::-1], legend=True, stacked=True, width=0.6, linewidth=0)

    ax.set_xticks(np.arange(0, 41, 4))
    ax.set_xticklabels(np.arange(1981, 2022, 4))
    ax.set_ylim(-18, 0)
    yticks = np.arange(-18, 0.1, 3)
    ax.set_yticks(yticks)
    # ylabels = [f"{100*x:.0f}%" for x in yticks]
    # ax.set_yticklabels(ylabels)
    ax.set_ylabel("Accumulated SPEI", fontsize=10)
    ax.tick_params(labelsize=10)
    ax.set_title(f"Drought Severity", fontsize=10, loc="left")

    ########################
    ax2 = fig.add_axes([0.65, 0.1, 0.2, 0.4])
    # color2 = ["#2878B5", "#F8AC8C"]
    color2 = ["#1681FC", "#FD7F1E"]
    data2.plot.bar(
        rot=0, ax=ax2, color=color2, width=0.6, linewidth=0)
    ax2.set_ylim(-4, 0)
    yticks = np.arange(-4, 0.1, 1)
    ax2.set_yticks(yticks)
    # ylabels = [f"{100*x:.0f}%" for x in yticks]
    # ax2.set_yticklabels(ylabels)
    # ax.set_ylabel("Area Percentage", fontsize=10)
    ax2.set_title(f"Drought Severity Comparison", fontsize=10, loc="left")

    # 图例
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, ncol=2, fontsize=8)

    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, ['Before 1999', 'After 1999'], fontsize=8)

    plt.savefig(rf'E:/SPEI_base/python MG/JPG/{title}.jpg',
                bbox_inches='tight')


area_plot(df_dd.iloc[:, ::-1], dd_cp, title=r"MG Grassland Drought Severity")
