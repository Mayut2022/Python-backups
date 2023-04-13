# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 11:36:24 2023

曲线拟合效果很差
拟合方法存疑，以及我认为不能采用此方法，是一个包含土壤湿度阈值的复杂变化
@author: MaYutong
"""

import warnings
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
from scipy.stats import linregress

from math import log
from sklearn.metrics import mean_squared_error

plt.rcParams['font.sans-serif'] = ['times new roman']  # 指定默认字体

warnings.filterwarnings("ignore")


# %%
# yr = np.arange(1983, 2021)

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
    inpath = (r"E:/LAI4g/data_MG/LAI_82_20_MG_SPEI0.5x0.5.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        # print(f.variables['lai'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        lai = (f.variables['lai'][:, 4:9, :, :])
        lai_gs = np.nanmean(lai, axis=1)
        lai_diff = lai_gs[1:, ]-lai_gs[:-1, ]
    return lai_diff


def read_nc2():
    inpath = (r"E:/SPEI_base/data/spei03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        spei = (f.variables['spei'][972:, :, :])
        spei = spei.reshape(39, 12, 30, 50)
        spei_gs = np.nanmean(spei[:, 4:9, :, :], axis=1)
        spei_diff = spei_gs[1:, ]-spei_gs[:-1, ]
    return spei_diff


# %%
lcc = read_lcc()
lai = read_nc()
spei = read_nc2()

# 转折前
# lai = lai[:18,]
# spei = spei[:18,]

# 转折后
lai = lai[18:, ]
spei = spei[18:, ]


#%% 随机点画图测试
def plot1(x, y, y_pred1, y_pred2, mse1, mse2, aic1, aic2):
    plt.figure(1, s=(10, 10), dpi=300)
    xlabel = np.arange(1983, len(x)+1983)
    plt.scatter(x, y, c="k")
    plt.plot(x, y_pred1, c="r", label="Linear")
    plt.plot(x, y_pred2, c="b", label="quadratic")
    
    plt.legend()
    plt.show()


def plot2(x, y, y_pred1, y_pred2, mse1, mse2, aic1, aic2):
    print(len(x), len(y))
    plt.figure(1, figsize=(10, 10), dpi=300)
    # xlabel = np.arange(2001, len(x)+2001)
    plt.scatter(x, y, c="k")
    plt.plot(x, y_pred1, c="r", label="Linear")
    plt.plot(x, y_pred2, c="b", label="quadratic")
    
    plt.legend()
    plt.show()
    


# %% AIC
def calculate_aic(n, mse, num_params):
    aic = n * log(mse) + 2 * num_params
    return aic


# %% 一次拟合和二次拟合
aic_lin = np.zeros((30, 50))
aic_qua = np.zeros((30, 50))
# n_wet = np.zeros((30, 50))
# n_dry = np.zeros((30, 50))
n = 0
for i in range(30):
    for j in range(50):
        a = lcc[i, j]
        if a == 130:
            y = lai[:, i, j].data
            x = spei[:, i, j].data
            if np.isnan(x).any() or np.isnan(y).any():
                aic_lin[i, j] = np.nan
                aic_qua[i, j] = np.nan
            else:
                # 一阶拟合
                z1 = np.polyfit(x, y, 1)
                p1 = np.poly1d(z1)
                y_pred1 = p1(x)
                mse1 = mean_squared_error(y, y_pred1)
                aic_lin[i, j] = calculate_aic(len(y), mse1, 1)

                # 二阶拟合
                z1 = np.polyfit(x, y, 2)
                p1 = np.poly1d(z1)
                y_pred2 = p1(x)
                mse2 = mean_squared_error(y, y_pred2)
                aic_qua[i, j] = calculate_aic(len(y), mse2, 1)
                
                if j%20==0:
                    n+=1
                    plot2(x, y, y_pred1, y_pred2, mse1, mse2, aic_lin[i, j], aic_qua[i, j])
        else:
            aic_lin[i, j] = np.nan
            aic_qua[i, j] = np.nan


# %% 两拟合比较
aic_com = np.zeros((30, 50))
for i in range(30):
    for j in range(50):
        a = aic_lin[i, j]
        b = aic_qua[i, j]
        if np.isnan(a).any() or np.isnan(b).any():
            aic_com[i, j]=np.nan
        else:
            if a<b:
                aic_com[i, j] = 100
            else:
                aic_com[i, j] = -5