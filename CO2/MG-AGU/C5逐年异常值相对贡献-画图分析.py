# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 16:05:26 2023

@author: MaYutong
"""
import warnings
import netCDF4 as nc
import matplotlib.dates as mdate
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


def read_nc_anom():
    inpath = (r"E:/CO2/python-AGU-MG/FittingC5MG_Anom_Con.nc")
    with nc.Dataset(inpath) as f:
        print(f.variables.keys())
        # lat = (f.variables['lat'][:])
        # lon = (f.variables['lon'][:])
        spei_a = (f.variables['spei_anom'][:])
        tmp_a = (f.variables['tmp_anom'][:])
        co2_a = (f.variables['co2_anom'][:])
        sos_a = (f.variables['sos_anom'][:])
        # gpp1, gpp2 = gpp[:18, ], gpp[18:, ]

    return spei_a, tmp_a, co2_a, sos_a


def read_nc_con():
    inpath = (r"E:/CO2/python-AGU-MG/FittingC5MG_Anom_Con.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        # lat = (f.variables['lat'][:])
        # lon = (f.variables['lon'][:])
        spei_c = (f.variables['spei_con'][:])
        tmp_c = (f.variables['tmp_con'][:])
        co2_c = (f.variables['co2_con'][:])
        sos_c = (f.variables['sos_con'][:])
        # gpp1, gpp2 = gpp[:18, ], gpp[18:, ]

    return spei_c, tmp_c, co2_c, sos_c

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


# %%
lcc = read_lcc()
var = ["spei", "tmp", "co2", "sos"]
spei_a, tmp_a, co2_a, sos_a = read_nc_anom()
spei_c, tmp_c, co2_c, sos_c = read_nc_con()
for x in var:
    exec(f"{x}_am = mask(130, {x}_a)")
    exec(f"{x}_cm = mask(130, {x}_c)")
