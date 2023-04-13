# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 11:34:04 2023

@author: MaYutong
"""


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

        spei = (f.variables['spei'][[1, 2], 80:, :, ])


# %% mask数组
def mask(x, data):
    lcc2 = np.array(lcc)

    np.place(lcc2, lcc2 != x, 1)
    np.place(lcc2, lcc2 == x, 0)

    spei_ma = np.empty((2, 40, 30, 50))
    spei_ma = np.ma.array(spei_ma)

    for i in range(2):
        for l in range(40):
            a = data[i, l, :, :]
            a = ma.masked_array(a, mask=lcc2, fill_value=-999)
            spei_ma[i, l, :, :] = a

    spei_ma_ave = np.nanmean(spei_ma, axis=(2, 3))

    return spei_ma_ave

# %% 计算每年遭受严重干旱的面积


def area(x, data):
    lcc2 = np.array(lcc)

    np.place(lcc2, lcc2 != x, 1)
    np.place(lcc2, lcc2 == x, 0)

    area = np.empty((2, 40))

    for i in range(2):
        for l in range(40):
            a = data[i, l, :, :]
            a = ma.masked_array(a, mask=lcc2, fill_value=-999)
            a = a.filled()
            a = a.reshape(1, 1500)
            a = np.squeeze(a)
            a = a[a != 1.e+20]
            bin = [-10, -1, 10]
            _ = pd.cut(a, bin, right=True)
            num = _.value_counts().sort_index()
            per = num/383
            area[i, l] = per[-1]

    return area

# %% mk突变检验


def mktest(inputdata):
    inputdata = np.array(inputdata)
    n = inputdata.shape[0]
    s = 0
    Sk = np.zeros(n)
    UFk = np.zeros(n)
    for i in range(1, n):
        for j in range(i):
            if inputdata[i] > inputdata[j]:
                s = s+1
            else:
                s = s+0
        Sk[i] = s
        E = (i+1)*(i/4)
        Var = (i+1)*i*(2*(i+1)+5)/72
        UFk[i] = (Sk[i] - E)/np.sqrt(Var)

    Sk2 = np.zeros(n)
    UBk = np.zeros(n)
    s = 0
    inputdataT = inputdata[::-1]
    for i in range(1, n):
        for j in range(i):
            if inputdataT[i] > inputdataT[j]:
                s = s+1
            else:
                s = s+0
        Sk2[i] = s
        E = (i+1)*(i/4)
        Var = (i+1)*i*(2*(i+1)+5)/72
        UBk[i] = -(Sk2[i] - E)/np.sqrt(Var)
    UBk2 = UBk[::-1]
    return UFk, UBk2


def tip(uf, ub):
    year = np.arange(1981, 2021, 1)
    sign = np.sign(uf[0]-ub[0])
    for yr, a, b in zip(year, uf, ub):
        if np.sign(a-b) == -sign:
            print(yr, a, b)
            sign = -sign


# %%
def plot(data1, data2, data3, title):
    fig, axs = plt.subplots(2, 2, figsize=(16, 8), dpi=150, sharex=True)
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9,
                        top=0.9, wspace=0.1, hspace=0.15)

    sea_str = ["Summer", "Autumn"]

    t = np.arange(1981, 2021)

    for i in range(2):
        # 渐变色柱状图
        # 归一化
        norm = plt.Normalize(-2, 2)  # 值的范围
        norm_values = norm(data1[i, :])
        map_vir = cm.get_cmap(name='bwr_r')
        colors = map_vir(norm_values)
        #axs[0, i].scatter(t, data1[ind, :], c='k', s=20) #
        axs[0, i].bar(t, data1[i, :], color=colors, label="SPEI03", zorder=2)

        axs[0, i].set_title(f"{sea_str[i]} SPEI03", fontsize=15, loc="left")
        axs[0, i].set_ylim(-2.2, 2.2)
        axs[0, i].set_yticks(np.arange(-2, 2.1, 1))
        axs[0, i].tick_params(labelsize=15)

    # axs[1, 0]干旱面积
    color = ["green", "orange"]
    width = 0.45  # the width of the bars
    multiplier = 0

    for i in range(2):
        num = spei_area[i, :]
        offset = width * multiplier
        rects = axs[1, 0].bar(t + offset, num, width, color=color[i],
                              alpha=0.5, label=sea_str[i])
        multiplier += 1
    axs[1, 0].set_yticks(np.arange(0, 1.1, 0.2))
    axs[1, 0].tick_params(labelsize=15)
    axs[1, 0].legend(fontsize=12)
    axs[1, 0].set_title(f"drought area percentage".title(), fontsize=15, loc="left")
    
    # axs[1, 1]mk突变检验
    uf1, ub1 = mktest(spei_ave[0, :])
    uf2, ub2 = mktest(spei_ave[1, :])
    axs[1, 1].plot(t, uf1, 'green', linestyle="-", label='UFk')
    axs[1, 1].plot(t, ub1, 'green', linestyle="--", label='UBk')
    axs[1, 1].scatter(t, uf1, color='green', s=20)
    axs[1, 1].scatter(t, ub1, color='green', s=20)
    axs[1, 1].plot(t, uf2, "orange", linestyle="-", label='UFk')
    axs[1, 1].plot(t, ub2, "orange", linestyle="--", label='UBk')
    axs[1, 1].set_ylim(-3.8, 3.8)
    axs[1, 1].set_yticks(np.arange(-3, 3.1, 1))
    axs[1, 1].scatter(t, uf2, color="orange", s=20)
    axs[1, 1].scatter(t, ub2, color="orange", s=20)
    axs[1, 1].tick_params(labelsize=15)
    axs[1, 1].legend(fontsize=12, ncol=2)
    axs[1, 1].set_title(f"mk trend test".title(), fontsize=15, loc="left")

    plt.suptitle(f"{title}", fontsize=20)
    # plt.savefig(rf'E:/SPEI_base/python MG/JPG/{title}.jpg',
    #             bbox_inches='tight')
    plt.show()


# %% spei 夏季、秋季分布图
read_nc()
read_nc2()

spei_ave = mask(130, spei)
spei_area = area(130, spei)

print(np.nanmax(spei_ave), np.nanmin(spei_ave))
print(np.nanmax(spei_area), np.nanmin(spei_area))
plot(spei_ave, spei_area, spei_ave, f"MG Grassland Drought Characters")

# %%
