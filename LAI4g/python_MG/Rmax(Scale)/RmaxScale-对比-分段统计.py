# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 21:32:56 2023

@author: MaYutong
"""

from collections import Counter

import netCDF4 as nc
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import xlsxwriter

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


def read_lcc():
    # global a1, a2, o1, o2
    inpath = (r"E:/ESA/ESA_LUC_2000_SPEI03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        # print(f.variables['lccs_class'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        lcc = (f.variables['lcc'][:])
    return lcc


# %% mask数组


def mask(x, data):
    sp = data.shape[0]
    lcc2 = np.array(lcc)

    np.place(lcc2, lcc2 != x, 1)
    np.place(lcc2, lcc2 == x, 0)

    spei_ma = np.empty((sp, 30, 50))
    spei_ma = np.ma.array(spei_ma)

    for i in range(sp):
        a = data[i, :, :]
        a = np.ma.masked_array(a, mask=lcc2)
        spei_ma[i, :, :] = a

    return spei_ma.data

# %%


def lcc_count(data):
    sp = data.shape
    data = data.reshape(1, sp[0]*sp[1]*sp[2])
    data = np.squeeze(data)

    lcc_c = Counter(data)
    lcc_c = pd.Series(lcc_c)
    lcc_c = lcc_c.sort_index()

    return lcc_c.iloc[:12]


# %%

def plot(data1, data2, data3, title):
    print(np.nanmax(data1), np.nanmin(data1))
    print(np.nanmax(data2), np.nanmin(data2))

    print(np.nanmax(data3), np.nanmin(data3))

    fig = plt.figure(figsize=(18, 3), dpi=500)
    fig.subplots_adjust(left=0.1, bottom=0.15, right=0.9,
                        top=0.9, wspace=0.2, hspace=0.2)

    yr_str = ["(a)1982-1999", "(b)2000-2020", "DIFF(b-a)"]

    scale = [str(x).zfill(2) for x in range(1, 13)]
    for j in range(1, 4):
        ax = fig.add_subplot(1, 3, j)
        if j == 3:
            data3 = data3/data1.sum()
            # 渐变色柱状图
            # 归一化
            norm = plt.Normalize(-0.15, 0.15)  # 值的范围
            norm_values = norm(data3)
            map_vir = cm.get_cmap(name='BrBG')
            colors = map_vir(norm_values)
            ax.axhline(0, color='k', linewidth=0.8)
            ax.bar(scale, data3, color=colors, width=0.8, edgecolor="k")
            ax.set_ylim(-0.06, 0.06)
            yticks = np.arange(-0.06, 0.061, 0.02)
            ax.set_yticks(yticks)
            ylabels = [f"{100*x:.0f}%" for x in yticks]
            ax.set_yticklabels(ylabels)
        else:
            data = eval(f"data{j}")
            ax.bar(scale, data, color='orange', alpha=0.5,
                   width=0.8, edgecolor='orange')
            ax.set_ylim(0, 0.2)
            yticks = np.arange(0, 0.21, 0.05)
            ax.set_yticks(yticks)
            ylabels = [f"{100*x:.0f}%" for x in yticks]
            ax.set_yticklabels(ylabels)

        ax.set_title(f"Sensitivity Scale {yr_str[j-4]}",
                     fontsize=15, loc="left", color="b")
        ax.tick_params(labelsize=15)
        # ax.grid(which="major", color="dimgray", linestyle='--', linewidth=0.3)
        if j == 1:
            ax.set_ylabel("Area Percentage", fontsize=20)
        elif j == 2:
            ax.set_xlabel("SPEI Time Scale", fontsize=20)

    # plt.suptitle(f'{title}', fontsize=25)
    plt.savefig(
        rf'E:/LAI4g/JPG_MG/Rmax(Scale)/{title}.jpg', bbox_inches='tight')
    plt.show()


# %%
lcc = read_lcc()
win = 30
inpath = rf"E:/LAI4g/data_MG/Rmax(Scale)/LAI_SPEI_MG_MonthWindows{win}.nc"
_, r_s, _ = read_nc(inpath)

r_s_mask = mask(130, r_s)

# mask部分
r_s_1 = r_s_mask[:90, :, :]
r_s_2 = r_s_mask[90:, :, :]

cou1 = lcc_count(r_s_1)
cou1 = cou1/cou1.sum()
cou2 = lcc_count(r_s_2)
cou2 = cou2/cou2.sum()

cou3 = cou2-cou1

# %%

plot(cou1, cou2, cou3, title=r"LAI RmaxScale Comparison")
