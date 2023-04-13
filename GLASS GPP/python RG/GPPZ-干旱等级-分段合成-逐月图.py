# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 20:35:19 2023

@author: MaYutong
"""
import warnings
import matplotlib.dates as mdate
import matplotlib.pyplot as plt

import netCDF4 as nc
import numpy as np
import pandas as pd
import xarray as xr

plt.rcParams['font.sans-serif'] = ['times new roman']  # 指定默认字体
warnings.filterwarnings("ignore")

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


def read_nc():
    global lat, lon
    inpath = (r"E:/GLASS-GPP/MG detrend Zscore/MG_GPP_Zscore_SPEI_0.5x0.5.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        # print(f.variables['GPP_Z'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        gpp = (f.variables['GPP_Z'][:, 4:9, :])
        gpp1, gpp2 = gpp[:18, ], gpp[18:, ]

        gpp11 = gpp1.reshape(18*5, 30, 50)
        gpp22 = gpp2.reshape(19*5, 30, 50)

    return gpp11, gpp22


def read_nc2():
    inpath = (r"E:/SPEI_base/data/spei03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        spei = (f.variables['spei'][972:-24, :])
        spei = spei.reshape(37, 12, 30, 50)
        spei_gsl = spei[:, 4:9, :].reshape(37*5, 30, 50)
        spei1, spei2 = spei_gsl[:90, :], spei_gsl[90:, :]
    return spei1, spei2


# %%
def read_nc12():
    inpath = (r"E:/SPEI_base/data/spei12_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        spei = (f.variables['spei'][972:-24, :])
        spei = spei.reshape(37, 12, 30, 50)
        spei_gsl = spei[:, 4:9, :].reshape(37*5, 30, 50)
        spei1, spei2 = spei_gsl[:90, :], spei_gsl[90:, :]
    return spei1, spei2


def read_nc01():
    inpath = (r"E:/SPEI_base/data/spei01_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        spei = (f.variables['spei'][972:-24, :])
        spei = spei.reshape(37, 12, 30, 50)
        spei_gsl = spei[:, 4:9, :].reshape(37*5, 30, 50)
        spei1, spei2 = spei_gsl[:90, :], spei_gsl[90:, :]
    return spei1, spei2

# %%


def t_xaxis():
    year = np.arange(1982, 2019)
    for i, yr in enumerate(year):
        t = pd.date_range(f'{yr}/05', periods=5, freq="MS")
        if i == 0:
            t_all = t
        else:
            t_all = np.hstack((t_all, t))

    return t_all
# %% mask数组


def mask(x, data):
    sp = data.shape
    lcc2 = np.array(lcc)

    np.place(lcc2, lcc2 != x, 1)
    np.place(lcc2, lcc2 == x, 0)

    spei_ma = np.empty((sp[0], 30, 50))
    spei_ma = np.ma.array(spei_ma)

    for l in range(sp[0]):
        a = data[l, :, :]
        a = np.ma.masked_array(a, mask=lcc2)
        spei_ma[l, :, :] = a

    spei_ma_ave = np.nanmean(spei_ma, axis=(1, 2))

    return spei_ma_ave

# %% mask数组, 统计干旱月份和对应的数据
# 筛选干旱与非干旱


def gpp_drought(x, gpp, spei):
    print(str(x))

    # 非干旱统计
    # 植被值
    gppm1 = gpp.copy()
    exec(f"gppm1[{x}]=np.nan")
    gppm_N = locals()['gppm1']
    gNm = mask(130, gppm_N)
    # 月份统计
    gppc1 = gpp.copy()
    exec(f"gppc1[{x}]=0")
    exec(f"gppc1[~({x})]=1")
    gppc_N = locals()['gppc1']
    gNc = mask(130, gppc_N)

    # 干旱统计
    # 植被值
    gppm2 = gpp.copy()
    exec(f"gppm2[~({x})]=np.nan")
    gppm_Y = locals()['gppm2']
    gYm = mask(130, gppm_Y)
    # 月份统计
    gppc2 = gpp.copy()
    exec(f"gppc2[~({x})]=0")
    exec(f"gppc2[{x}]=1")
    gppc_Y = locals()['gppc2']
    gYc = mask(130, gppc_Y)

    return gNm, gNc, gYm, gYc  # 依次为：非干旱的植被平均值，非干旱月数；干旱的植被平均值，干旱月数

# %%


def com_plot(data1, data2, title):
    global xlabels
    print(np.nanmax(data1), np.nanmin(data1))
    print(np.nanmax(data2), np.nanmin(data2))

    fig = plt.figure(1, figsize=(18, 9), dpi=500)
    fig.subplots_adjust(left=0.05, bottom=0.1,
                        right=0.95, top=0.90, hspace=0.15)
    axes = fig.subplots(2, 1, sharex=True)

    t = np.arange(185)
    xticks = t[2::5]
    # tt = t_xaxis()
    # xlabels = tt[2::5]

    # 趋势线拟合
    # z1 = np.polyfit(t, data2, 1)
    # p1 = np.poly1d(z1)
    # data2_pred = p1(t)
    # print(data2_pred)

    ###########
    ax = axes[0]
    ax.axvline(90, color="b", linewidth=2, zorder=2)
    ax.text(90, 0.92, "1999(turning point)", c='b', fontsize=15)
    ax.bar(t, data1, color="orange", width=0.8, alpha=0.8)
    ax.set_xticks(xticks)
    # ax.set_xticklabels(xlabels)
    # ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y'))
    ax.set_xlim(-5, 190)
    ax.set_ylim(0, 0.1)
    yticks = np.arange(0, 1.1, 0.2)
    ax.set_yticks(yticks)
    ylabels = [f"{100*x:.0f}%" for x in yticks]
    ax.set_yticklabels(ylabels)
    ax.tick_params(labelsize=15)
    ax.set_title("(a) Drought Area Percentage", fontsize=15, loc="left")

    ###########
    ax2 = axes[1]
    ax2.axvline(90, color="b", linewidth=2, zorder=2)
    ax2.text(90, 1.45, "1999(turning point)", c='b', fontsize=15)
    ax2.axhline(y=0, c="k", linestyle="-", linewidth=1)
    ax2.axhline(gYm1_ave, xmin=0, xmax=(90/185), c="k", linestyle="--", linewidth=2)
    ax2.axhline(gYm2_ave, xmin=(90/185), xmax=1, c="r", linestyle="--", linewidth=2)
    ax2.bar(t, data2, color='green', alpha=0.8)
    ax2.set_ylim(-1.7, 1.7)
    yticks = np.arange(-1.5, 1.6, 0.5)
    ax2.set_yticks(yticks)
    ax2.tick_params(labelsize=15)
    ax2.set_title("(b) GPP Z-score", fontsize=15, loc="left")
    # ax2.plot(t, data2_pred, c="b", linestyle="--", linewidth=1, zorder=3)
    
    # ax2.legend()

    plt.suptitle(f'{title}', fontsize=20)
    plt.savefig(rf'E:/GLASS-GPP/JPG MG/Comparison3/{title}.jpg',
                bbox_inches='tight')
    plt.show()


# %%
def PLOT(data1, data2, name):
    # data3 = data2-data1

    title = rf"{name} Drought (SPEI03)"
    # title = rf"{name} Drought (SPEI12 01)"
    print("\n", title)

    # print('5 percentile is: ', np.nanpercentile(data1, 5))
    # print('95 percentile is: ', np.nanpercentile(data2, 95), "\n")

    # print("DIFF")
    # print('5 percentile is: ', np.nanpercentile(data3, 5))
    # print('95 percentile is: ', np.nanpercentile(data3, 95), "\n")

    com_plot(data1, data2, title)


# %%
lcc = read_lcc()
gpp1, gpp2 = read_nc()
spei1, spei2 = read_nc2()
# spei1, _ = read_nc12()
# _, spei2 = read_nc01()

# %%
# lev_d = [
#     "(spei<-0.5)&(spei>=-1)", "(spei<-1)&(spei>=-1.5)", "(spei<-1.5)&(spei>=-2)", "(spei<-2)"]
# lev_n = ["Mild", "Moderate", "Severe", "Extreme"]

# for lev1, name in zip(lev_d[:], lev_n[:]):
#     _, _, gYm1, gYc1 = gpp_drought(lev1, gpp1, spei1)
#     _, _, gYm2, gYc2 = gpp_drought(lev1, gpp2, spei2)
#     gYm = np.hstack((gYm1, gYm2))
#     gYc = np.hstack((gYc1, gYc2))
#     PLOT(gYc, gYm, name)


# %%
lev_d = ["(spei<-0.5)&(spei>=-2)", "(spei<-2)"]
lev_n = ["MMS", "Extreme"]

for lev1, name in zip(lev_d[:1], lev_n[:1]):
    _, _, gYm1, gYc1 = gpp_drought(lev1, gpp1, spei1)
    _, _, gYm2, gYc2 = gpp_drought(lev1, gpp2, spei2)
    gYm1_ave, gYm2_ave = np.nanmean(gYm1), np.nanmean(gYm2)
    gYm = np.hstack((gYm1, gYm2))
    gYc = np.hstack((gYc1, gYc2))
    PLOT(gYc, gYm, name)
    