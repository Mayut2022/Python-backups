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
def ds_ave(data, codes):
    labels = ["Extreme", "Severe", "Moderate", "Mild"]
    df_ds = pd.DataFrame(dict(spei=data, codes=codes))
    grouped = df_ds["spei"].groupby(df_ds["codes"])
    ds = grouped.mean()
    if len(ds) == 4:
        ds[-1] = np.nan
    else:
        ds = ds[1:]
    ds.index = labels
    ds.name = "Drought Severity"
    return ds


def area_cut(data):
    level = dict(dry1="x <= -2",
                 dry2="x>-2 and x<=-1.5",
                 dry3="x>-1.5 and x<=-1",
                 dry4="x>-1 and x<=-0.5")
    bin = [-np.inf, -2, -1.5, -1, -0.5]
    labels = ["Extreme", "Severe", "Moderate", "Mild"]
    cut = pd.cut(data, bin, right=True, labels=labels)
    data_count = cut.value_counts()/4596
    data_count.name = 'Area Percentage'

    ds = ds_ave(data, cut.codes)

    # print(data_count, "\n")

    return ds, data_count


# %%
read_nc()
spei_ma = mask(130, spei)
spei_mn = spei_ma.reshape(40, 12, 30, 50)


# %% 某一年test

# spei_yr = spei_mn[21, :, ]
# spei_ds = spei_yr[spei_yr != -9999]
# ds, area = area_cut(spei_ds)


# %%
df_ds = pd.Series()
df_area = pd.Series()

for yr in range(40):
    spei_yr = spei_mn[yr, :, ]

    # 强度
    spei_ds = spei_yr[spei_yr != -9999]
    ds, area = area_cut(spei_ds)
    df_ds = pd.concat([df_ds, ds])
    df_area = pd.concat([df_area, area])

# %%


def multi_index():
    global year_index
    year = np.arange(1981, 2021)
    year_index = []
    for i, x in enumerate(year):
        for j in range(4):
            year_index.append(x)



multi_index()
df_ds.index = [year_index, df_ds.index]

df_ds_usk = df_ds.unstack(level=-1)

df_area.index = [year_index, df_area.index]
df_area_usk = df_area.unstack()


# %%
def comparison(data):
    global data1, data2
    data1 = data.iloc[:19, :]
    data2 = data.iloc[19:, :]
    
    data1 = data1.mean()
    data2 = data2.mean()
    

    data1 = pd.concat([data1, data2])
    index = ["Before", "Before", "Before", "Before",
             "After", "After", "After", "After"]
    data1.index = [index, data1.index]
    
    area_cp = data1.unstack(level=0)
    area_cp = area_cp.iloc[::-1, ::-1] ###交换次序
    return area_cp


area_cp = comparison(df_area_usk)

# %%


def area_plot(data, data2, title):
    global children, handles, labels
    fig = plt.figure(figsize=(12, 4), dpi=1080)

    ax = fig.add_axes([0.2, 0.1, 0.4, 0.4])

    color = ["#ffff68", "orange", "red", "#c40000"]

    ax.axvline(18, color="b", linewidth=1, zorder=2)
    ax.text(18, 0.7, "1999(turning point)", c='b')
    # linewidth控制间隙
    data.plot.bar(
        rot=0, ax=ax, color=color[::-1], legend=True, stacked=True, width=0.6, linewidth=0)

    ax.set_xticks(np.arange(0, 41, 4))
    ax.set_xticklabels(np.arange(1981, 2022, 4))
    ax.set_ylim(0, 0.8)
    yticks = np.arange(0, 0.81, 0.2)
    ax.set_yticks(yticks)
    ylabels = [f"{100*x:.0f}%" for x in yticks]
    ax.set_yticklabels(ylabels)
    ax.set_ylabel("Area Percentage", fontsize=10)
    ax.tick_params(labelsize=10)
    ax.set_title(f"(c) Drought Area", fontsize=10, loc="left")

    ########################
    ax2 = fig.add_axes([0.65, 0.1, 0.2, 0.4])
    # color2 = ["#2878B5", "#F8AC8C"]
    color2 = ["#1681FC", "#FD7F1E"]
    data2.plot.bar(
        rot=0, ax=ax2, color=color2, width=0.6, linewidth=0)
    ax2.set_ylim(0, 0.2)
    yticks = np.arange(0, 0.21, 0.05)
    ax2.set_yticks(yticks)
    ylabels = [f"{100*x:.0f}%" for x in yticks]
    ax2.set_yticklabels(ylabels)
    # ax.set_ylabel("Area Percentage", fontsize=10)
    ax2.set_title(f"(d) Drought Area Comparison", fontsize=10, loc="left")
    
    ###图例
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, ncol=2, fontsize=8)
    
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, ['1981-1999', '2000-2020'], fontsize=8)
    
    plt.savefig(rf'E:/SPEI_base/python MG/JPG/{title}.jpg',
                bbox_inches='tight')


area_plot(df_area_usk, area_cp, title=r"MG Grassland Drought Area")
