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
    inpath = (r"E:/SPEI_base/data/spei03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        spei = (f.variables['spei'][960:, :, :])
        spei = spei.reshape(40, 12, 30, 50)
        spei_gsl = spei[:, 4:9, :]
        spei_gsl = np.nanmean(spei_gsl, axis=1)  # 81-20
        return spei_gsl


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
            
# %% MK 突变检验

def mk_plot(fig, data):
    ax = fig.add_axes([0.55, 0.1, 0.3, 0.4])
    
    t = np.arange(1981, 2021)
    uf, ub = mktest(data)
    ax.plot(t, uf, 'r', label='UFk', linewidth=1)
    ax.plot(t, ub, 'b', label='UBk', linewidth=1)
    ax.scatter(t, uf, color='r', s=5)
    ax.scatter(t, ub, color='b', s=5)
    ax.axhline(y=1.96, c="k", linestyle="--", linewidth=1)
    ax.axhline(y=-1.96, c="k", linestyle="--", linewidth=1)
    ax.set_ylim(-3.8, 3.8)
    ax.set_yticks(np.arange(-3, 3.1, 1.5))
    plt.legend(loc="upper right")
    ax.set_title(f"MK Test", fontsize=10, loc="left")
    
    return ax

# %%


def plot(data1, title):
    fig = plt.figure(figsize=(12, 4), dpi=500)

    t = np.arange(1981, 2021)

    # 滑动相关
    t2 = np.arange(1985, 2016)
    data1_roll = pd.Series(data1)
    data1_roll = data1_roll.rolling(10).mean().dropna()

    # 趋势线拟合
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

    ax.set_title(f"Growing Season SPEI03", fontsize=10, loc="left")
    ax.set_ylim(-2.2, 2.2)
    ax.set_yticks(np.arange(-2, 2.1, 1))
    ax.tick_params(labelsize=10)

    ax2 = mk_plot(fig, data1)

    # fig.suptitle(f"{title}", fontsize=20)
    plt.savefig(rf'E:/SPEI_base/python MG/JPG/{title}.jpg',
                bbox_inches='tight')
    plt.show()


# %% spei 夏季、秋季分布图
read_nc()
spei = read_nc2()

spei_ave = mask(130, spei)

print(np.nanmax(spei_ave), np.nanmin(spei_ave))

plot(spei_ave, f"MG Grassland Drought Characters2 GSL")
