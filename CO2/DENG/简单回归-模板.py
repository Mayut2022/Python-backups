# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 21:51:55 2022

@author: MaYutong
"""
# %%
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np

import pandas as pd
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm

import xarray as xr

# %%
def read_nc(inpath):
    global gpp, lat2, lon2
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())

        gpp = (f.variables['GPP'][:])
        lat2 = (f.variables['lat'][:])
        lon2 = (f.variables['lon'][:])

    return gpp


def sif_xarray2(band1):
    mn = np.arange(12)
    sif = xr.DataArray(band1, dims=['mn', 'y', 'x'], coords=[mn, lat2, lon2])

    sif_MG = sif.loc[:, 40:55, 100:125]
    return np.array(sif_MG)


def exact_data3():
    for yr in range(1982, 2019):
        inpath = rf"/mnt/e/GLASS-GPP/Month RG/GLASS_GPP_RG_SPEI_0.5X0.5_{yr}.nc"
        data = read_nc(inpath)

        data_MG = sif_xarray2(data)
        data_MG = data_MG.reshape(1, 12, 30, 50)

        if yr == 1982:
            data_all = data_MG
        else:
            data_all = np.vstack((data_all, data_MG))

    return data_all

# %%


def read_nc2():
    inpath = r"/mnt/e/CRU/Q_DATA_CRU-GLEAM/PRE_global_81_20.nc"
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())

        lat_g = (f.variables['lat'][:])
        lon_g = (f.variables['lon'][:])
        pre = (f.variables['pre'][:])
        # 480(输入data) -> 月 年

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

        pre_mn = mn_yr(pre)

        def sif_xarray(band1):
            mn = np.arange(12)
            yr = np.arange(40)
            sif = xr.DataArray(band1, dims=['mn', 'yr', 'y', 'x'], coords=[
                               mn, yr, lat_g, lon_g])

            sif_MG = sif.loc[:, :, 40:55, 100:125]
            return np.array(sif_MG)

    pre_mn_MG = sif_xarray(pre_mn)
    return pre_mn_MG

# %%


def read_nc3():
    inpath = r"/mnt/e/CO2/DENG/CO2_81_13_RG_SPEI0.5x0.5.nc"
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())

        lat3 = (f.variables['lat'][:])
        lon3 = (f.variables['lon'][:])
        co2 = (f.variables['co2'][:])

    ###########
    def region(data):
        t = np.arange(396)
        data_global = xr.DataArray(data, dims=['t', 'y', 'x'], coords=[
                                   t, lat3, lon3])  # 原SPEI-base数据
        data_rg = data_global.loc[:, 40:55, 100:125]

        return data_rg

    ###########
    def mn_yr(data):
        q_mn = []
        for mn in range(12):
            q_ = []
            for yr in range(33):
                q_.append(data[mn])
                mn += 12
            q_mn.append(q_)

        q_mn = np.array(q_mn)
        return q_mn

    co2 = region(co2)
    co2_MG = mn_yr(co2)

    return co2_MG


# %%
gpp_MG = exact_data3()
pre = read_nc2()
pre = pre[:, 1:38, :, :]
co2 = read_nc3()

# %% fit statsmodels包
x1 = pre[6, 1:33, 10, 10]
x2 = co2[6, 1:, 10, 10]
y = gpp_MG[1:33, 6, 10, 10]

df = pd.DataFrame(dict(y=y, x1=x1, x2=x2))

fit = sm.formula.ols('y~x1+x2', data=df).fit()  # df格式/数组格式均可
print(fit.params)
print(fit.summary())

# y拟合值
y_fit = fit.predict()
# 等同于fit.fittedvalues
# R-squared 评分
fit.rsquared

# F检验，检验模型是否显著
fit.fvalue
fit.f_pvalue
# t检验，检验自变量是否显著
fit.tvalues
fit.pvalues

# %% plot绘图
fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

ax.plot(x1, y, "o", label="data")
ax.plot(x1, y_fit, "r--.", label="OLS")
ax.legend(loc="best")
