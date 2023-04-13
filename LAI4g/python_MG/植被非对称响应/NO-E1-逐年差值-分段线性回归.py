# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 14:05:22 2023
该方法存在问题，首先不能用原值进行研究，原值研究存在一些问题
将前后分段，干湿分别提取，通过检验的点较小，可能受限于样本数或者是变化关系不是简单的线性关系
考虑用LAI变化速率进行研究
@author: MaYutong
"""


import warnings
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
from scipy.stats import linregress

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

## 转折前
lai = lai[:18,]
spei = spei[:18,]

## 转折后
# lai = lai[18:,]
# spei = spei[18:,]


# %% 单点测试
n_grass = np.zeros((30, 50))
n_all = np.zeros((30, 50))
n_wet = np.zeros((30, 50))
n_dry = np.zeros((30, 50))

for i in range(30):
    for j in range(50):
        a = lcc[i, j]
        if a == 130:
            n_grass[i, j] = 1

            b = lai[:, i, j]
            c = spei[:, i, j]

            ind = c > 0  # wet
            b_w, c_w = b[ind], c[ind]
            b_d, c_d = b[~ind], c[~ind]

            s, inter, r, p, _ = linregress(c, b)
            if p <= 0.05:
                n_all[i, j] = 1
            # print("All")
            # print("slope:", s, "intercept:", inter,
            #       "\n", "corr:", r, "p-value", p, "\n")

            s, inter, r, p, _ = linregress(c_w, b_w)
            if p <= 0.05:
                n_wet[i, j] = 1
            # print("Wet")
            # print("slope:", s, "intercept:", inter,
            #       "\n", "corr:", r, "p-value", p, "\n")

            s, inter, r, p, _ = linregress(c_d, b_d)
            if p <= 0.05:
                n_dry[i, j] = 1
            # print("Dry")
            # print("slope:", s, "intercept:", inter, "\n", "corr:", r, "p-value", p)
        else:
            pass

# %%
var_all = ["n_grass", "n_all", "n_wet", "n_dry"]

for var in var_all:
    plt.figure(1, dpi=500)
    exec(f"plt.pcolormesh({var}, cmap='Reds')")
    plt.title(var, fontsize=15)
    # plt.colorbar(shrink=0.75, orientation='horizontal')
    plt.show()