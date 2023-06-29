# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 21:09:30 2023

@author: MaYutong
"""

import cmaps

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

plt.rcParams['font.sans-serif'] = ['times new roman']  # 指定默认字体

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


def read_LAI():
    global lat, lon
    # inpath = (r"E:/LAI4g/data_MG/LAI_Detrend_Anomaly_82_20_MG_SPEI0.5x0.5.nc")
    inpath = (r"E:/LAI4g/data_MG/LAI_82_20_MG_SPEI0.5x0.5.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        # print(f.variables['lai'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        lai = (f.variables['lai'][:, 4:9, :, :])
        lai_gs = np.nanmean(lai, axis=1)
        lai_diff = lai_gs[1:,]-lai_gs[:-1,]

        lai_std = np.nanstd(lai_diff, axis=0)
        
        lai_std_all = []
        for i in range(38):
            lai_std_all.append(lai_std)
            
        lai_std_all = np.array(lai_std_all)
            
    return lai_diff, lai_std_all




# %% mask数组


def read_spei_1():
    inpath = (r"E:/SPEI_base/data/spei03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        spei = (f.variables['spei'][960:, :, :])
        spei = spei.reshape(40, 12, 30, 50)
        spei_gsl = spei[:, 4:9, :]
        spei_gs = np.nanmean(spei_gsl, axis=1)
        spei_diff = spei_gs[1:,]-spei_gs[:-1,]
        
        spei_std = np.nanstd(spei_diff, axis=0)
        
        spei_std_all = []
        for i in range(39):
            spei_std_all.append(spei_std)
            
        spei_std_all = np.array(spei_std_all)

    return spei_diff[1:], spei_std_all[1:]


# %%


def t_xaxis():
    year = np.arange(1981, 2021)
    for i, yr in enumerate(year):
        t = pd.date_range(f'{yr}/05', periods=5, freq="MS")
        tt = []
        for j, x in enumerate(t):
            tt.append(x.strftime("%Y-%m"))
        if i == 0:
            tt_all = tt
        else:
            tt_all = tt_all+tt

    return tt_all

# %%


def plot_all(data1, data2, data3, data4, title):
    fig = plt.figure(1, figsize=(5, 5), dpi=500)

    fig.subplots_adjust(left=0.05, bottom=0.1, right=0.95,
                        top=0.9, wspace=None, hspace=0.2)

    t = np.arange(38)
    t_date = np.arange(1983, 2021)

    axes = fig.subplots(2, 1, sharex=True)
    ax1, ax2 = axes[0], axes[1]
    # ax1.axvline(17, color="gray", linewidth=1.5, zorder=3)
    # ax1.text(17, 3, "1999(drought turning point)", c='gray', fontsize=15)
    # ax2.axvline(17, color="gray", linewidth=1.5, zorder=3)
    # ax2.text(17, 0.65, "1999(drought turning point)", c='gray', fontsize=15)

    # ax1
    ax1.axhline(y=0, c="gray", linestyle="--", linewidth=1)
    ax1.axhline(y=data2[0], c="b", linestyle="--", linewidth=1.5, label="1.0 SPEI SD")
    ax1.axhline(y=-data2[0], c="r", linestyle="--", linewidth=1.5, label="-1.0 SPEI SD")
    # 渐变色柱状图
    # 归一化
    norm = plt.Normalize(-0.2, 0.2)  # 值的范围
    norm_values = norm(data1)
    cmap = cmaps.MPL_bwr_r
    map_vir = cm.get_cmap(cmap)
    colors = map_vir(norm_values)

    ax1.bar(t, data1, color=colors, zorder=2, alpha=0.8)

    ax1.tick_params(labelsize=15)
    ax1.set_ylim(-2.5, 2.5)
    ax1.set_yticks(np.arange(-2, 2.1, 1))

    ax1.set_title("(c) SPEI03", fontsize=15, loc="left")
    # ax1.set_ylabel("SPEI03 Detrend", fontsize=15)
    ax1.legend(loc="upper left")
    
    # ax2
    ax2.axhline(0, color="gray", linewidth=1, linestyle='--')
    ax2.scatter(t, data3, c='k', s=10, zorder=2)
    ax2.plot(t, data3, c='k', linewidth=1.5, zorder=2)
    ax2.plot(t, 0.5*data4, c='b', label="0.5 LAI SD", linewidth=1, zorder=1)
    ax2.plot(t, -0.5*data4, c='r', label="-0.5 LAI SD", linewidth=1, zorder=1)

    ax2.tick_params(labelsize=15)
    ax2.set_xlim(-2, 41)
    ax2.set_xticks(t[::5])
    ax2.set_xticklabels(t_date[::5])
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_yticks(np.arange(-0.5, 0.51, 0.25))

    ax2.set_title("(d) LAI", fontsize=15, loc="left")
    # ax2.set_ylabel("LAI Detrend Anomaly", fontsize=15)
    ax2.legend(loc="upper left")
    # plt.xlabel("years", fontsize=15)

    # plt.suptitle(f'{title}', fontsize=20)
    plt.savefig(rf'E:/LAI4g/python_MG/抵抗力和恢复力2/JPG/{title}.jpg',
                bbox_inches='tight')
    plt.show()


# %%
lcc = read_lcc()
lai, lai_std = read_LAI()
spei, spei_std = read_spei_1()


# %%
spei1_t = spei[:, 20, 32]
spei_std_t = spei_std[:, 20, 32]
lai_t = lai[:, 20, 32]
lai_std_t = lai_std[:, 20, 32]

plot_all(spei1_t, spei_std_t, lai_t, lai_std_t, title=f"Sample Interannual Increment")



