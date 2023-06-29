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
    inpath = (r"E:/LAI4g/data_MG/LAI_Anomaly_82_20_MG_SPEI0.5x0.5.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        # print(f.variables['lai'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        lai = (f.variables['lai'][:, 4:9, :, :])
        lai_gs = np.nanmean(lai, axis=1)
        lai_diff = lai_gs[1:,]-lai_gs[:-1,]
        lai_7 = (f.variables['lai'][:, 6, :, :])
            
    # return lai_gs
    # return lai_diff
    return lai_7

# %%


def read_spei():
    inpath = (r"E:/SPEI_base/data/spei03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        spei = (f.variables['spei'][960:, :, :])
        spei = spei.reshape(40, 12, 30, 50)
        spei_gsl = spei[:, 4:9, :]
        spei_gs = np.nanmean(spei_gsl, axis=1)
        spei_diff = spei_gs[1:,]-spei_gs[:-1,]
        
        spei_7 = spei[1:, 6, :]
    
    # return spei_gs[1:]
    # return spei_diff[1:]
    return spei_7



#%%


def read_WUE():
    inpath = (r"E:/LAI4g/data_MG/WUE_LAI_ET(T)_82_20_MG_SPEI0.5x0.5.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        wue = (f.variables['WUE2'][:])  ####1982-2020
        # wue_gsl = wue[:, 4:9, :]
        # wue_gs = np.nanmean(wue_gsl, axis=1)
        wue_7 = wue[:, 6, :]
        
    return wue_7


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
        a = ma.masked_array(a, mask=lcc2)
        spei_ma[i, :, :] = a

    spei_ma_ave = np.nanmean(spei_ma, axis=(1, 2))

    return spei_ma_ave


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


def plot_all(data1, data2, data3, title):
    fig = plt.figure(1, figsize=(7, 9), dpi=500)

    fig.subplots_adjust(left=0.05, bottom=0.1, right=0.95,
                        top=0.95, wspace=None, hspace=0.15)

    t = np.arange(39)
    t_date = np.arange(1982, 2021)

    axes = fig.subplots(3, 1, sharex=True)
    ax1, ax2, ax3 = axes[0], axes[1], axes[2]
    ax1.axvline(17, color="orange", linewidth=1.5, zorder=3)
    ax1.text(17, 3, "1999(drought turning point)", c='orange', fontsize=15)
    ax2.axvline(17, color="orange", linewidth=1.5, zorder=3)
    ax2.text(17, 0.65, "1999(drought turning point)", c='orange', fontsize=15)
    ax3.axvline(17, color="orange", linewidth=1.5, zorder=3)
    # ax2.text(17, 0.65, "1999(drought turning point)", c='orange', fontsize=15)
    

    # ax1
    ax1.set_title("July", fontsize=15, loc="left")
    ax1.axhline(y=0, c="gray", linestyle="--", linewidth=1)
    ax1.axhline(y=1, c="b", linestyle="--", linewidth=1.5)
    ax1.axhline(y=-1, c="r", linestyle="--", linewidth=1.5)
    # 渐变色柱状图
    # 归一化
    norm = plt.Normalize(-0.2, 0.2)  # 值的范围
    norm_values = norm(data1)
    cmap = cmaps.MPL_bwr_r
    map_vir = cm.get_cmap(cmap)
    colors = map_vir(norm_values)
    ax1.bar(t, data1, color=colors, label="SPEI03", zorder=2, alpha=0.8)

    ax1.tick_params(labelsize=15)
    ax1.set_ylim(-3.5, 3.5)
    ax1.set_yticks(np.arange(-3, 3.1, 1))
    # ax1.set_ylabel("SPEI03 Oringinal", fontsize=15)
    ax1.set_ylabel("SPEI03", fontsize=15)

    # ax2
    ax2.set_title("July", fontsize=15, loc="left")
    ax2.axhline(0, color="gray", linewidth=1, linestyle='--')
    ax2.scatter(t, data2, c='k', s=10, zorder=2)
    ax2.plot(t, data2, c='k', label="LAI", linewidth=1.5, zorder=2)
    
    ax2.tick_params(labelsize=15)
    ax2.set_xlim(-2, 41)
    ax2.set_xticks(t[::5])
    ax2.set_xticklabels(t_date[::5])
    ax2.set_ylim(-0.8, 0.8)
    ax2.set_yticks(np.arange(-0.75, 0.76, 0.25))
    ax2.set_ylabel("LAI Anomaly", fontsize=15)
    ax2.legend()
    
    # ax3
    ax3.set_title("July", fontsize=15, loc="left")
    ax3.axhline(0, color="gray", linewidth=1, linestyle='--')
    ax3.scatter(t, data3, c='purple', s=10, zorder=2)
    ax3.plot(t, data3, c='purple', label="WUE", linewidth=1.5, zorder=2)
    ax3.tick_params(labelsize=15)
    ax3.set_ylim(-0.02, 0.15)
    ax3.set_ylabel("WUE(LAI/T)", fontsize=15)
    ax3.legend()

    plt.suptitle(f'{title}', fontsize=20)
    # plt.savefig(rf'E:/LAI4g/python_MG/抵抗力和恢复力/JPG/{title}.jpg',
    #             bbox_inches='tight')
    plt.show()


# %%
lcc = read_lcc()
lai = read_LAI()
spei = read_spei()
wue = read_WUE()


# %%
for r in range(30):
    for c in range(50):
        if lcc[r, c]!=130:
            pass
        elif r%3==0 and c%5==0:
            spei_t = spei[:, r, c]
            lai_t = lai[:, r, c]
            wue_t = wue[:, r, c]
            title = rf"SITE r:{r} c:{c} lat:{lat[r]} lon:{lon[c]}"
            plot_all(spei_t, lai_t, wue_t, title=title)
