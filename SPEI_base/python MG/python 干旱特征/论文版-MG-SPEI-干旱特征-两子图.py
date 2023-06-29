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
from scipy.stats import linregress
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
    
    
# %%


def read_nc3():
    global lat, lon
    # inpath = (r"E:/LAI4g/data_MG/LAI_82_20_MG_SPEI0.5x0.5.nc")
    inpath = (r"E:/LAI4g/data_MG/LAI_Zscore_82_20_MG_SPEI0.5x0.5.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        # print(f.variables['lai'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        lai = (f.variables['lai'][:, 4:9, :, :])
        lai_gs = np.nanmean(lai, axis=1)

    return lai_gs


# %% mask数组
def mask(x, data):
    lcc2 = np.array(lcc)
    
    sp = data.shape[0]

    np.place(lcc2, lcc2 != x, 1)
    np.place(lcc2, lcc2 == x, 0)

    spei_ma = np.empty((sp, 30, 50))
    spei_ma = np.ma.array(spei_ma)

    for l in range(sp):
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
    ax = fig.add_axes([0.2, 0.1, 0.3, 0.4])
    
    t = np.arange(1981, 2021)
    uf, ub = mktest(data)
    ax.plot(t, uf, 'r', label='UFk', linewidth=1.5)
    ax.plot(t, ub, 'b', label='UBk', linewidth=1.5)
    ax.scatter(t, uf, color='r', s=5)
    ax.scatter(t, ub, color='b', s=5)
    ax.axhline(y=1.96, c="k", linestyle="--", linewidth=1) ###95%显著性水平
    ax.axhline(y=-1.96, c="k", linestyle="--", linewidth=1)
    ax.set_ylim(-3.8, 3.8)
    ax.set_yticks(np.arange(-3, 3.1, 1.5))
    plt.legend(loc="upper right", fontsize=8)
    ax.set_title(f"(g) SPEI03 MK Test", fontsize=12, loc="left")
    
    return ax

# %%


def plot(data1, data2, title):
    fig = plt.figure(figsize=(12, 4), dpi=500)

    t_date = pd.date_range(f'1980', periods=41, freq="YS")
    tt = []
    for j, x in enumerate(t_date):
        tt.append(x.strftime("%Y"))
    t2 = np.arange(40)


    ax = fig.add_axes([0.55, 0.1, 0.3, 0.4])
    ##########################
    
    ax.axhline(y=0, c="k", linestyle="--")
    # ax.axvline(18, color="gray", linewidth=1.5, zorder=3)
    # ax.text(18, -1.8, "1999(drought turning point)", c='b', fontsize=10)
    ax.text(12, 1.6, "LAI Trend: 0.017/a", c='b', fontsize=12, weight='bold')
    ax.text(30, 1.6, "CORR: 0.63", c='k', fontsize=12, weight='bold')
    # 渐变色柱状图
    # 归一化
    norm = plt.Normalize(-1, 1)  # 值的范围
    norm_values = norm(data2)
    cmap = cmaps.BlueWhiteOrangeRed_r[18:239]
    map_vir = cm.get_cmap(cmap)
    colors = map_vir(norm_values)
    
    ax.bar(t2, data2, color=colors, edgecolor='none', label="SPEI03", zorder=1)
    ax.tick_params(labelsize=10)
    ax.set_ylabel("SPEI03", fontsize=10)
    ax.set_ylim(-2, 2)
    ax.set_title(f"(h) Growing Season Mean", fontsize=12, loc="left")
    ax.set_xticks(np.arange(-1, 41, 10))
    ax.set_xticklabels(np.arange(1980, 2021, 10))
    
    ##########################
    # # 趋势线拟合
    z1 = np.polyfit(t2[1:], data1, 1)
    p1 = np.poly1d(z1)
    data1_pred = p1(t2[1:])
    
    ax2 = ax.twinx()
    # ax2.plot(t2[1:], data1_pred, c="b", linestyle="--", linewidth=1.5, zorder=3)
    ax2.scatter(t2[1:], data1, c='k', s=10, zorder=2)
    ax2.plot(t2[1:], data1, c='k', label="LAI Z-Score", linewidth=1.5, zorder=2)
    ax2.tick_params(labelsize=10)
    ax2.set_ylabel("LAI Z-Score", fontsize=10)
    ax2.set_ylim(-1.5, 1.5)
    
    ################## label
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1+handles2, labels1+labels2, ncol=1, fontsize=8, loc="lower left")

    ax2 = mk_plot(fig, data2)

    # fig.suptitle(f"{title}", fontsize=20)
    # plt.savefig(rf'E:/SPEI_base/python MG/JPG/{title}.jpg',
    #             bbox_inches='tight')
    # plt.savefig(rf'F:/000000 论文图表-正文/{title}.jpg',
    #             bbox_inches='tight')
    plt.show()


# %% spei 夏季、秋季分布图
read_nc()
spei = read_nc2()

spei_ave = mask(130, spei)

lai = read_nc3()
lai_mk = mask(130, lai)

print(np.nanmax(spei_ave), np.nanmin(spei_ave))
print(np.nanmax(lai_mk), np.nanmin(lai_mk))

plot(lai_mk, spei_ave, f"MG Grassland Drought Characters3 GSL")


#%%
yr=np.arange(1982, 2021, 1)
s,_,r, p,_  = linregress(yr, lai_mk)
print(s, p)