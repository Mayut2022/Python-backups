# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 12:39:01 2023

@author: MaYutong
"""

import netCDF4 as nc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# plt.rcParams['font.sans-serif'] = ['times new roman']  # 指定默认字体

plt.rcParams['font.sans-serif']=['simsun']
plt.rcParams['axes.unicode_minus'] = False

import warnings
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
    inpath = (r"E:/LAI4g/data_MG/LAI_Detrend_Zscore_82_20_MG_SPEI0.5x0.5.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        # print(f.variables['lai'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        lai = (f.variables['lai'][:, 4:9, :, :])
        lai_gsl = lai.reshape(39*5, 30, 50)

    return lai_gsl


# %% 土壤湿度


def MG(data):
    t = np.arange(1, 481, 1)
    l = np.arange(1, 5, 1)
    spei_global = xr.DataArray(data, dims=['t', 'l', 'y', 'x'], coords=[
                               t, l, lat2, lon2])  # 原SPEI-base数据
    spei_rg1 = spei_global.loc[:, :, 40:55, 100:125]
    spei_rg1 = np.array(spei_rg1)
    return spei_rg1


def read_nc2():
    global lat2, lon2
    # inpath = rf"E:/GLDAS Noah/DATA_RG/SM_81_20_ORINGINAL_SPEI0.5x0.5.nc"
    inpath = rf"E:/GLDAS Noah/DATA_RG/SM_Zscore_81_20_SPEI0.5x0.5.nc"
    with nc.Dataset(inpath, mode='r') as f:
        # print(f.variables.keys())
        lat2 = (f.variables['lat'][:])
        lon2 = (f.variables['lon'][:])
        sm = (f.variables['sm'][:, :, :])
        sm_MG = MG(sm)
        sm = sm_MG.reshape(40, 12, 4, 30, 50)
        sm_gsl = sm[:, 2:7, :]
        sm_gsl = sm_gsl.reshape(200, 4, 30, 50)
    return sm_gsl[5:, 0, ]


# %% 滑动相关计算


def roll_corr(data1, data2, windows):
    corr = np.zeros((195, 30, 50))
    for i in range(30):
        for j in range(50):
            a = pd.Series(data1[:, i, j], name="LAI")
            b = pd.Series(data2[:, i, j], name="SM")
            if np.isnan(a).any() or np.isnan(b).any():
                corr[:, i, j] = np.nan
            else:
                df = pd.concat([a, b], axis=1)
                r = df.rolling(windows).corr().iloc[1::2, 0]
                corr[:, i, j] = r

    return corr

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

    spei_ma_ave = np.nanmean(spei_ma, axis=(1, 2))

    return spei_ma_ave


# %%生成新的nc文件
def CreatNC(windows, data1):
    new_NC = nc.Dataset(
        rf"E:/LAI4g/data_MG/moving corr/LAI_SPEI_detrend_Win{windows}.nc", 'w', format='NETCDF4')

    time = np.arange(195)

    new_NC.createDimension('time', 195)
    new_NC.createDimension('lat', 30)
    new_NC.createDimension('lon', 50)

    var = new_NC.createVariable('corr', 'f', ("time", "lat", "lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))

    new_NC.variables['corr'][:] = data1
    new_NC.variables['lat'][:] = lat
    new_NC.variables['lon'][:] = lon

    # 最后记得关闭文件
    new_NC.close()

# %%


def t_xaxis():
    year = np.arange(1982, 2021)
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

def plot_Grass(data1, win, sig, title):
    
    wh, w = int(win/2), win.copy() ####1/2滑动窗口，滑动窗口
    _ = [np.nan]*wh
    _ = np.array(_)
    data1= np.hstack((data1, _))

    fig = plt.figure(1, figsize=(16, 4), dpi=500)

    fig.subplots_adjust(left=0.05, bottom=0.1, right=0.95,
                        top=0.9, wspace=None, hspace=0.12)

    t_date = t_xaxis()
    t = np.arange(195)
    ax = fig.subplots(1)
    ax.axhline(y=sig, c="r", linestyle="--")
    ax.axvline(85, color="b", linewidth=2, zorder=2)
    ax.text(85, 0.58, "1999(turning point)", c='b', fontsize=15)
    # "#ffb432"Grass;  "#285000" Forest
    ax.scatter(t[wh:], data1[w:], c='k', s=10)
    ax.plot(t[wh:], data1[w:], c='k', label="Grassland", linewidth=1)
    ax.tick_params(labelsize=15)
    ax.set_ylabel("Correlation coefficient", fontsize=15)
    ax.set_ylim(-0.3, 0.65)
    ax.set_yticks(np.arange(-0.3, 0.61, 0.15))
    ax.set_xlim(-10, 205)
    ax.set_xticks(t[::5])
    ax.set_xticklabels(t_date[::5], rotation=60)
    # ax.xaxis.set_major_formatter(mdate.DateFormatter('%y-%m'))

    plt.legend(loc='upper right', fontsize=15)
    plt.xlabel("years", fontsize=15)
    plt.suptitle(f'{title}', fontsize=20)
    # plt.savefig(rf'E:/LAI4g/JPG_MG2/{title}.jpg',
    #             bbox_inches='tight')
    plt.show()



# %% 数据读取
lcc = read_lcc()
lai = read_nc()
sm1 = read_nc2()

win_all = np.arange(10, 51, 10)
sig_all = [0.5494, 0.3783, 0.3061, 0.2638, 0.2353]

for win, sig in zip(win_all[2:], sig_all[2:]):
    r = roll_corr(lai, sm1, win)
    # CreatNC(win, r)
    r_g = mask(130, r)
    print("草地 LAI-SM", np.nanmax(r_g), np.nanmin(r_g))
    # plot_Grass(r_g, win, sig, title=rf"相关 LAI-SM同期 (Win={win}) ")
    plot_Grass(r_g, win, sig, title=rf"相关 LAI-SM超前2月 (Win={win}) ")
    
