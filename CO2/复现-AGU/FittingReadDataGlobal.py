# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 17:09:19 2022

@author: MaYutong
"""
# %%
import cftime
import netCDF4 as nc
import numpy as np
import os
import pandas as pd
import xarray as xr

# %%
lat_g = np.linspace(-89.75, 89.75, 360)
lon_g = np.linspace(-179.75, 179.75, 720)


# %% 读取因变量Y, 自变量X


class Y():

    def read(self, inpath, var):
        # print(inpath)
        with nc.Dataset(inpath) as f:
            # print(f.variables.keys())
            # lat = (f.variables['lat'][:])
            # lon = (f.variables['lon'][:])
            data = (f.variables[f'{var}'][:]).data
        return data

# %% 480(输入data) -> 月 年


class MnYr:
    def mn_yr(self, data):
        sp = data.shape[0]
        tmp_mn = []
        for mn in range(12):
            tmp_ = []
            for yr in range(int(sp/12)):
                tmp_.append(data[mn])
                mn += 12
            tmp_mn.append(tmp_)

        tmp_mn = np.array(tmp_mn)

        return tmp_mn


# %% 类的组合

class Rd:
    y = Y()
    mnyr = MnYr()

    def read(self, inpath, var):
        data1 = self.y.read(inpath, var)
        if var in ["pre", "sr", "tmp"]:
            data2 = self.mnyr.mn_yr(data1)
            return data2
        else:
            return data1

# %%


def y_GPP():
    for yr in range(1982, 2019):
        inpath = rf"/mnt/e/GLASS-GPP/Month RG/GLASS_GPP_RG_SPEI_0.5X0.5_{yr}.nc"
        data = Y().read(inpath, "GPP")
        data_MG = XR().rg(data)
        data_MG = data_MG.reshape(1, 12, 30, 50)

        if yr == 1982:
            # print(locals())
            data_all = data_MG
        else:
            data_all = np.vstack((data_all, data_MG))

    return data_all

######################
# %%


def data_read():
    rd = Rd()

    inpath_sif = r"/mnt/e/Gosif_Monthly/data_Global/GOSIF_01_20_SPEI0.5X0.5.nc"
    sif = rd.read(inpath_sif, "sif")
    #################

    inpath_pre = r"/mnt/e/CRU/Q_DATA_CRU-GLEAM/PRE_global_81_20.nc"
    inpath_tmp = r"/mnt/e/CRU/TMP_DATA/TMP_CRU_ORIGINAL_81_20.nc"
    inpath_vpd = r"/mnt/e/CRU/VAP_DATA/vpd_CRU_MONTH_81_20.nc"
    inpath_sr = r"/mnt/e/GLDAS Noah/DATA_GLOBAL/SR_81_20_global/sr_ORINGINAL_SPEI_0.5X0.5.nc"
    inpath_co2_CT = r"/mnt/e/CO2/CT2019B/molefractions/XCO2 Global/XCO2_00_18_SPEI0.5x0.5.nc"
    pre = rd.read(inpath_pre, "pre")
    tmp = rd.read(inpath_tmp, "tmp")
    tmp[tmp == 9.96921e+36] = np.nan
    vpd = Y().read(inpath_vpd, "vpd")
    sr = rd.read(inpath_sr, "sr")
    co2_CT = Y().read(inpath_co2_CT, "co2")

    print("sif, pre, tmp, vpd, sr, co2_CT")
    return sif, pre, tmp, vpd, sr, co2_CT


sif, pre, tmp, vpd, sr, co2_CT = data_read()

# %%


class Data_exact:
    input = "input the year like: 1981, 1982, etc. "

    def exact2(self, data, sy, ey, sm, em):
        print("SIF, from 2001 to 2020")
        data = data[sy-2001:ey-2001+1, sm-1:em, :, :]
        return data

    def exact3(self, data, sy, ey, sm, em):
        print("pre, tmp, vpd, sr from 81 to 20")
        data = data[sm-1:em, sy-1981:ey-1981+1, :, :]
        return data

    def exact4(self, data, sy, ey, sm, em):
        print("co2 Carbon Tracker, from 2000 to 2018")
        data = data[sy-2000:ey-2000+1, sm-1:em, :, :]
        return data


def data_exact(sy, ey, sm, em):
    global de

    de = Data_exact()
    sif_de = de.exact4(sif, sy, ey, sm, em)
    co2_ct_de = de.exact4(co2_CT, sy, ey, sm, em)

    var = ["pre", "tmp", "vpd", "sr"]
    for x in var:
        exec(f"{x}_de = de.exact3({x}, sy, ey, sm, em)", globals())

    print("sif, pre, tmp, vpd, sr, co2_CT")
    return sif_de, pre_de, tmp_de, vpd_de, sr_de, co2_ct_de


def data_time(a, b, c, d):
    global sy, ey, sm, em
    sy, ey, sm, em = a, b, c, d


# %%
if __name__ == "__main__":
    # 读取全部数据
    sif, pre, tmp, vpd, sr, co2 = data_read()

    data_time(2001, 2018, 4, 10)
    sif_de, pre_de, tmp_de, vpd_de, sr_de, co2_ct_de = data_exact(2001, 2018, 4, 10)


# %%
