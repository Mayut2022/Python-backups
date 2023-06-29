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
    inpath = (r"E:/LAI4g/data_MG/LAI_Anomaly_82_20_MG_SPEI0.5x0.5.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        # print(f.variables['lai'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        lai = (f.variables['lai'][:, 4:9, :, :])
        lai_gs = np.nanmean(lai, axis=1)

        lai_std = np.nanstd(lai_gs, axis=0)

    return lai_gs, lai_std

# %%


def read_spei_1():
    inpath = (r"E:/SPEI_base/data/spei03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        spei = (f.variables['spei'][960:, :, :])
        spei = spei.reshape(40, 12, 30, 50)
        spei_gsl = spei[:, 4:9, :]
        spei_gs = np.nanmean(spei_gsl, axis=1)

    return spei_gs[1:]


def read_spei_2():
    inpath = (r"E:/SPEI_base/data/spei03_MG_detrend.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        spei = (f.variables['spei'][:])
        spei_gsl = spei[4:9, :]
        spei_gs = np.nanmean(spei_gsl, axis=0)
    return spei_gs[1:]

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
    fig = plt.figure(1, figsize=(5, 5), dpi=500)

    fig.subplots_adjust(left=0.05, bottom=0.1, right=0.95,
                        top=0.9, wspace=None, hspace=0.2)

    t = np.arange(39)
    t_date = np.arange(1982, 2021)

    axes = fig.subplots(2, 1, sharex=True)
    ax1, ax2 = axes[0], axes[1]
    # ax1.axvline(17, color="gray", linewidth=1.5, zorder=3)
    # ax1.text(17, 3, "1999(drought turning point)", c='gray', fontsize=15)
    # ax2.axvline(17, color="gray", linewidth=1.5, zorder=3)
    # ax2.text(17, 0.65, "1999(drought turning point)", c='gray', fontsize=15)

    # ax1
    ax1.axhline(y=0, c="gray", linestyle="--", linewidth=1)
    # ax1.axhline(y=1, c="b", linestyle="--", linewidth=1.5)
    # ax1.axhline(y=-1, c="r", linestyle="--", linewidth=1.5)
    # 渐变色柱状图
    # 归一化
    norm = plt.Normalize(-0.2, 0.2)  # 值的范围
    norm_values = norm(data1)
    cmap = cmaps.MPL_bwr_r
    map_vir = cm.get_cmap(cmap)
    colors = map_vir(norm_values)

    ax1.bar(t, data1, color=colors, label="SPEI03", zorder=2, alpha=0.8)

    ax1.tick_params(labelsize=15)
    ax1.set_ylim(-2.5, 2.5)
    ax1.set_yticks(np.arange(-2, 2.1, 1))

    ax1.set_title("(a) SPEI03", fontsize=15, loc="left")
    # ax1.set_ylabel("SPEI03 Detrend", fontsize=15)

    # ax2
    ax2.axhline(0, color="gray", linewidth=1, linestyle='--')
    ax2.scatter(t, data2, c='k', s=10, zorder=2)
    ax2.plot(t, data2, c='k', label="LAI", linewidth=1.5, zorder=2)
    # ax2.plot(t, data3, c='b', label="0.5 LAI SD", linewidth=1, zorder=1)
    # ax2.plot(t, -data3, c='r', label="-0.5 LAI SD", linewidth=1, zorder=1)

    ax2.tick_params(labelsize=15)
    ax2.set_xlim(-2, 41)
    ax2.set_xticks(t[::5])
    ax2.set_xticklabels(t_date[::5])
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_yticks(np.arange(-0.5, 0.51, 0.25))

    ax2.set_title("(b) LAI Anomaly", fontsize=15, loc="left")
    # ax2.set_ylabel("LAI Detrend Anomaly", fontsize=15)
    # ax2.legend(loc="upper left")
    # plt.xlabel("years", fontsize=15)

    # plt.suptitle(f'{title}', fontsize=20)
    plt.savefig(rf'E:/LAI4g/python_MG/抵抗力和恢复力2/JPG/{title}.jpg',
                bbox_inches='tight')
    plt.show()


# %%
lcc = read_lcc()
lai, lai_std = read_LAI()
spei1 = read_spei_1()
# spei1 = read_spei_2()

# %%
spei1_t = spei1[:, 20, 32]
lai_t = lai[:, 20, 32]
lai_t_std = np.array([lai_std[20, 32]]*39)

plot_all(spei1_t, lai_t, lai_t_std, title=f"Sample Annual")


# # %%
# year = np.arange(1982, 2021)
# spei1_t0 = spei1_t[:-1].data
# spei1_t0 = np.insert(spei1_t0, 0, np.nan)
# spei1_t2 = spei1_t[1:].data
# spei1_t2 = np.append(spei1_t2, np.nan)

# lai_t0 = lai_t[:-1].data
# lai_t0 = np.insert(lai_t0, 0, np.nan)
# lai_t2 = lai_t[1:].data
# lai_t2 = np.append(lai_t2, np.nan)

# data = dict(YEAR=year, SPEIt0=spei1_t0, SPEIt1=spei1_t, SPEIt2=spei1_t2,
#             LAIt0=lai_t0, LAIt1=lai_t, LAIt2=lai_t2, LAIstd=lai_t_std)
# df = pd.DataFrame(data)


# #%% 从干旱出发，筛选干旱
# ind1 = df["SPEIt1"]<-1
# ind2 = df["SPEIt2"]>-1

# df["SPEIt1"][~ind1] = np.nan
# df["SPEIt2"][~ind2] = np.nan
# df = df.dropna(axis=0, how="all", subset=["SPEIt1"])

# #%%
# a = df["SPEIt2"].isnull()
# num = []
# b = 0
# for i, x in zip(range(len(df)), a):
#     if x:
#         print(i, x)
#         num.append(b)
#     else:
#         num.append(b)
#         b+=1

# df["NUM"] = num

# #%%
# df2 = pd.DataFrame()
# for name, group in df.groupby('NUM'):
#     # print(group, "\n")
#     # print(len(group))
#     if len(group)>1:
#         test = group
#         test["SPEIt0"].iloc[-1] = test["SPEIt0"].iloc[0]
#         test["LAIt0"].iloc[-1] = test["LAIt0"].iloc[0]
#         test["SPEIt1"].iloc[-1] = test["SPEIt1"].mean()
#         test["LAIt1"].iloc[-1] = test["LAIt1"].mean()
#         test = test.dropna()
#         df2 = pd.concat([df2, test])
#     else:
#         df2 = pd.concat([df2, group])


# #%%
# df2["SPEI dn"] = df2["SPEIt1"]-df2["SPEIt0"]
# df2["SPEI up"] = df2["SPEIt2"]-df2["SPEIt1"]

# df2["LAI dn"] = df2["LAIt1"]-df2["LAIt0"]
# df2["LAI up"] = df2["LAIt2"]-df2["LAIt1"]

# #%% 再筛选植被
# ind1 = abs(df2["LAI dn"])>=df2["LAIstd"]
# ind2 = abs(df2["LAI up"])>=df2["LAIstd"]

# df2["LAI dn"][~ind1] = np.nan
# df2["LAI up"][~ind2] = np.nan
# df2 = df2.dropna(axis=0, how="any", subset=["LAI dn", "LAI up"])


# df_all = df2.iloc[:, [0, 9, 10, 11, 12]]




