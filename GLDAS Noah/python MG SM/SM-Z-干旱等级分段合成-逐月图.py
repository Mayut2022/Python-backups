# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 16:25:53 2023

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


def mn_yr(data):
    tmp_mn = []
    for mn in range(12):
        tmp_ = []
        for yr in range(40):
            tmp_.append(data[mn])
            mn += 12
        tmp_mn.append(tmp_)

    tmp_mn = np.array(tmp_mn)

    return tmp_mn


def MG(data):
    t = np.arange(1, 481, 1)
    l = np.arange(1, 5, 1)
    spei_global = xr.DataArray(data, dims=['t', 'l', 'y', 'x'], coords=[
                               t, l, lat2, lon2])  # 原SPEI-base数据
    spei_rg1 = spei_global.loc[:, :, 40:55, 100:125]
    spei_rg1 = np.array(spei_rg1)
    return spei_rg1


def read_nc():

    def read_smz(inpath):
        global lat2, lon2
        with nc.Dataset(inpath, mode='r') as f:
            # print(f.variables.keys())
            sm_z = (f.variables['sm_z'][:])
            lat2 = (f.variables['lat'][:])
            lon2 = (f.variables['lon'][:])
        return sm_z

    for i in range(1, 13):
        inpath = rf"E:/GLDAS Noah/DATA_RG/Zscore_Anomaly/SM_81_20_SPEI0.5x0.5_Zscore_Anomaly_month{i}.nc"
        sm_z = read_smz(inpath)
        if i == 1:
            sm_z_all = sm_z
        else:
            sm_z_all = np.vstack((sm_z_all, sm_z))

    sm_z_all_MG = MG(sm_z_all)
    sm_z_all_MG = sm_z_all_MG.reshape(12, 40, 4, 30, 50)
    sm_gsl = sm_z_all_MG[4:9, :].reshape(5*40, 4, 30, 50, order='F')
    sm_gsl1, sm_gsl2 = sm_gsl[:95, ], sm_gsl[95:, ]
    return sm_gsl1, sm_gsl2


# %%

def read_nc2():
    global lat, lon
    inpath = (r"E:/SPEI_base/data/spei03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        spei = (f.variables['spei'][960:, :])
        spei = spei.reshape(40, 12, 30, 50)
        spei_gsl = spei[:, 4:9, :].reshape(40*5, 30, 50)
        spei1, spei2 = spei_gsl[:95, :], spei_gsl[95:, :]
    return spei1, spei2


# %%


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


def sm_drought(x, sm, spei):
    print(str(x))

    # 非干旱统计
    # 植被值
    smm1 = sm.copy()
    exec(f"smm1[{x}]=np.nan")
    smm_N = locals()['smm1']
    gNm = mask(130, smm_N)
    # 月份统计
    smc1 = sm.copy()
    exec(f"smc1[{x}]=0")
    exec(f"smc1[~({x})]=1")
    smc_N = locals()['smc1']
    gNc = mask(130, smc_N)

    # 干旱统计
    # 植被值
    smm2 = sm.copy()
    exec(f"smm2[~({x})]=np.nan")
    smm_Y = locals()['smm2']
    gYm = mask(130, smm_Y)
    # 月份统计
    smc2 = sm.copy()
    exec(f"smc2[~({x})]=0")
    exec(f"smc2[{x}]=1")
    smc_Y = locals()['smc2']
    gYc = mask(130, smc_Y)

    return gNm, gNc, gYm, gYc  # 依次为：非干旱的植被平均值，非干旱月数；干旱的植被平均值，干旱月数


# %%
def com_plot(data1, data2, title):
    

    fig = plt.figure(1, figsize=(18, 9), dpi=500)
    fig.subplots_adjust(left=0.05, bottom=0.1,
                        right=0.95, top=0.90, hspace=0.15)
    axes = fig.subplots(2, 1, sharex=True)

    t = np.arange(200)
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
    ax.axvline(95, color="b", linewidth=2, zorder=2)
    ax.text(95, 0.92, "1999(turning point)", c='b', fontsize=15)
    ax.bar(t, data1, color="orange", width=0.8, alpha=0.8)
    ax.set_xticks(xticks)
    # ax.set_xticklabels(xlabels)
    # ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y'))
    ax.set_xlim(-5, 205)
    ax.set_ylim(0, 1)
    yticks = np.arange(0, 1.1, 0.2)
    ax.set_yticks(yticks)
    ylabels = [f"{100*x:.0f}%" for x in yticks]
    ax.set_yticklabels(ylabels)
    ax.tick_params(labelsize=15)
    ax.set_title("(a) Drought Area Percentage", fontsize=15, loc="left")

    ###########
    ax2 = axes[1]
    ax2.axvline(95, color="b", linewidth=2, zorder=2)
    ax2.text(95, 2.7, "1999(turning point)", c='b', fontsize=15)
    ax2.axhline(y=0, c="k", linestyle="-", linewidth=1)
    ax2.axhline(gYm1_ave, xmin=0, xmax=(95/200),
                c="k", linestyle="--", linewidth=2)
    ax2.axhline(gYm2_ave, xmin=(95/200), xmax=1,
                c="r", linestyle="--", linewidth=2)
    ax2.bar(t, data2, color='brown', alpha=0.8)
    ax2.set_ylim(-3.1, 3.1)
    yticks = np.arange(-3, 3.1, 1)
    ax2.set_yticks(yticks)
    ax2.tick_params(labelsize=15)
    ax2.set_title("(b) SM Z-score", fontsize=15, loc="left")
    # ax2.plot(t, data2_pred, c="b", linestyle="--", linewidth=1, zorder=3)

    # ax2.legend()

    plt.suptitle(f'{title}', fontsize=20)
    plt.savefig(rf'E:/GLDAS Noah/JPG_MG/Comparison3/{title}.jpg',
                bbox_inches='tight')
    plt.show()


# %%
def PLOT(data1, data2, name, i):
    
    title = rf"{name} Drought Layer{i+1} SMZ (SPEI03)"
    print("\n", title)
    
    print(np.nanmax(data1), np.nanmin(data1))
    print(np.nanmax(data2), np.nanmin(data2))

    com_plot(data1, data2, title)
    
# %%
lcc = read_lcc()
sm1, sm2 = read_nc()
spei1, spei2 = read_nc2()


# %%
lev_d = [
    "(spei<-0.5)&(spei>=-1)", "(spei<-1)&(spei>=-1.5)", "(spei<-1.5)&(spei>=-2)", "(spei<-2)"]
lev_n = ["Mild", "Moderate", "Severe", "Extreme"]

for lev1, name in zip(lev_d[:], lev_n[:]):
    for layer in range(4):
        _, _, gYm1, gYc1 = sm_drought(lev1, sm1[:, layer,], spei1)
        _, _, gYm2, gYc2 = sm_drought(lev1, sm2[:, layer,], spei2)
        gYm1_ave, gYm2_ave = np.nanmean(gYm1), np.nanmean(gYm2)
        gYm = np.hstack((gYm1, gYm2))
        gYc = np.hstack((gYc1, gYc2))
        PLOT(gYc, gYm, name, layer)