# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 21:09:30 2023

@author: MaYutong
"""


import warnings
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
import xlsxwriter

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

    return lai_gs

# %%


def read_spei_1():
    inpath = (r"E:/SPEI_base/data/spei03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        spei = (f.variables['spei'][960:, :, :])
        spei = spei.reshape(40, 12, 30, 50)
        spei_gsl = spei[:, 4:9, :]
        spei_gs = np.nanmean(spei_gsl, axis=1)

    return spei_gs[1:, ]


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
def one_sample(r, c):
    spei_t = spei[:, r, c]
    lai_t = lai[:, r, c]
    lai_t_std = np.array([lai_t.data.std()]*39)
    if np.isnan(spei_t).any() or np.isnan(lai_t).any():
        df_all = pd.DataFrame()
    else:
        #######
        year = np.arange(1982, 2021)
        spei_t0 = spei_t[:-1].data
        spei_t0 = np.insert(spei_t0, 0, np.nan)
        spei_t2 = spei_t[1:].data
        spei_t2 = np.append(spei_t2, np.nan)

        lai_t0 = lai_t[:-1].data
        lai_t0 = np.insert(lai_t0, 0, np.nan)
        lai_t2 = lai_t[1:].data
        lai_t2 = np.append(lai_t2, np.nan)


        data = dict(YEAR=year, SPEIt0=spei_t0, SPEIt1=spei_t, SPEIt2=spei_t2,
                    LAIt0=lai_t0, LAIt1=lai_t, LAIt2=lai_t2, LAIstd=lai_t_std*0.5)
        df = pd.DataFrame(data)


        # 从干旱出发，筛选干旱
        ind1 = df["SPEIt1"] < -0.59
        ind2 = df["SPEIt2"] > -0.59

        df["SPEIt1"][~ind1] = np.nan
        df["SPEIt2"][~ind2] = np.nan
        df = df.dropna(axis=0, how="all", subset=["SPEIt1"])

        #########
        a = df["SPEIt2"].isnull()
        num = []
        b = 0
        for i, x in zip(range(len(df)), a):
            if x:
                # print(i, x)
                num.append(b)
            else:
                num.append(b)
                b += 1

        df["NUM"] = num

        #######
        df2 = pd.DataFrame()
        for name, group in df.groupby('NUM'):
            # print(group, "\n")
            # print(len(group))
            if len(group) > 1:
                test = group
                test["SPEIt0"].iloc[-1] = test["SPEIt0"].iloc[0]
                test["LAIt0"].iloc[-1] = test["LAIt0"].iloc[0]
                test["SPEIt1"].iloc[-1] = test["SPEIt1"].mean()
                test["LAIt1"].iloc[-1] = test["LAIt1"].mean()
                test = test.dropna()
                df2 = pd.concat([df2, test])
            else:
                df2 = pd.concat([df2, group])



        df2["SPEI dn"] = df2["SPEIt1"]-df2["SPEIt0"]
        df2["SPEI up"] = df2["SPEIt2"]-df2["SPEIt1"]

        df2["LAI dn"] = df2["LAIt1"]-df2["LAIt0"]
        df2["LAI up"] = df2["LAIt2"]-df2["LAIt1"]

        
        ind1 = abs(df2["LAI dn"])>=df2["LAIstd"]
        ind2 = abs(df2["LAI up"])>=df2["LAIstd"]

        df2["LAI dn"][~ind1] = np.nan
        df2["LAI up"][~ind2] = np.nan
        df2 = df2.dropna(axis=0, how="any", subset=["LAI dn", "LAI up"])


        df_all = df2.iloc[:, [0, 9, 10, 11, 12]]

    return df_all


# %%
lcc = read_lcc()
lai = read_LAI()
spei = read_spei_1()

# df = pd.DataFrame()
# for r in range(30):
#     for c in range(50):
#         if lcc[r, c] == 130:
#             df_one = one_sample(r, c)
#             df = pd.concat([df, df_one])
#         else:
#             pass


# %%

# df = df.sort_values(by="YEAR")
# df = df.dropna(axis=0)

#### SPEI<-1干旱，SPEI>-1恢复，不筛选植被
# df.to_excel("Sample.xlsx", index=False)

#### SPEI<-1干旱，SPEI>-1恢复，植被以0.5SD作为筛选标准
# df.to_excel("Sample2.xlsx", index=False)

# %%

# #%% 保存 excel数据

# dfs = {'Sig Up':df_wet, 'Sig Down':df_dry}
# writer = pd.ExcelWriter('LAI Sig Change2.xlsx', engine='xlsxwriter')

# for sheet_name in dfs.keys():
#     dfs[sheet_name].to_excel(writer, sheet_name=sheet_name, index=False)

# writer.save()
