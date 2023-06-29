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
            if var=="spei":
                data = (f.variables[f'{var}'][960:]).data
                data[data==1e+30] = np.nan
            else:
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
        if var in ["pre", "sr", "tmp", "co2", "pet", "spei"]:
            data2 = self.mnyr.mn_yr(data1)
            return data2
        else:
            return data1

# %%


def y_GPP():
    for yr in range(1982, 2019):
        inpath = rf"E:/GLASS-GPP/Month Global SPEI0.5x0.5/GLASS_GPP_{yr}.nc"
        data = Y().read(inpath, "GPP")
        data_G = data.reshape(1, 12, 360, 720)

        if yr == 1982:
            data_all = data_G
        else:
            data_all = np.vstack((data_all, data_G))

    return data_all

######################
# %%


def data_read():
    gpp = y_GPP()
    ##############
    rd = Rd()

    inpath_sif = r"E:/Gosif_Monthly/data_Global/GOSIF_00_20_SPEI0.5X0.5.nc"
    sif = rd.read(inpath_sif, "sif")
    #################

    inpath_pre = r"E:/CRU/Q_DATA_CRU-GLEAM/PRE_global_81_20.nc"
    inpath_pet = r"E:/CRU/PET_DATA/pet_CRU_ORIGINAL_81_20.nc"
    inpath_tmp = r"E:/CRU/TMP_DATA/TMP_CRU_ORIGINAL_81_20.nc"
    inpath_vpd = r"E:/CRU/VAP_DATA/vpd_CRU_MONTH_81_20.nc"
    inpath_sr = r"E:/GLDAS Noah/DATA_GLOBAL/SR_81_20_global/sr_ORINGINAL_SPEI_0.5X0.5.nc"
    inpath_co2 = r"E:/CO2/DENG/CO2_81_18_Global_Prolong_SPEI0.5x0.5.nc"
    inpath_spei = r"E:/SPEI_base/data/spei03.nc"

    pre = rd.read(inpath_pre, "pre")
    pet = rd.read(inpath_pet, "pet")
    pet[pet == 9.96921e+36] = np.nan
    tmp = rd.read(inpath_tmp, "tmp")
    tmp[tmp == 9.96921e+36] = np.nan
    vpd = Y().read(inpath_vpd, "vpd")
    sr = rd.read(inpath_sr, "sr")
    co2 = rd.read(inpath_co2, "co2")
    spei = rd.read(inpath_spei, "spei")

    print("gpp, sif, pre, pet, tmp, vpd, sr, co2, spei")
    return gpp, sif, pre, pet, tmp, vpd, sr, co2, spei


gpp, sif, pre, pet, tmp, vpd, sr, co2, spei = data_read()

# %%


class Data_exact:
    input = "input the year like: 1981, 1982, etc. "

    def exact1(self, data, sy, ey, sm, em):
        print("GPP, from 1982 to 2018")
        data = data[sy-1982:ey-1982+1, sm-1:em, :, :]
        return data

    def exact2(self, data, sy, ey, sm, em):
        print("SIF, from 2000 to 2020")
        data = data[sy-2000:ey-2000+1, sm-1:em, :, :]
        return data

    def exact3(self, data, sy, ey, sm, em):
        print("pre, pet, tmp, vpd, sr, co2, spei from 81 to 20")
        data = data[sm-1:em, sy-1981:ey-1981+1, :, :]
        return data

def data_time(a, b, c, d):
    global sy, ey, sm, em
    sy, ey, sm, em = a, b, c, d


def data_exact(sy, ey, sm, em):
    global de
    data_time(sy, ey, sm, em)

    de = Data_exact()
    gpp_de = de.exact1(gpp, sy, ey, sm, em)
    sif_de = de.exact2(sif, sy, ey, sm, em)

    var = ["pre", "pet", "tmp", "vpd", "sr", "co2", "spei"]
    for x in var:
        exec(f"{x}_de = de.exact3({x}, sy, ey, sm, em)", globals())

    print("gpp, sif, pre, pet, tmp, vpd, sr, co2, spei")
    return gpp_de, sif_de, pre_de, pet_de, tmp_de, vpd_de, sr_de, co2_de, spei_de


# %%
if __name__ == "__main__":
    # 读取全部数据
    # gpp, sif, pre, pet, tmp, vpd, sr, co2, spei = data_read()

    gpp_de, sif_de, pre_de, pet_de, tmp_de, vpd_de, sr_de, co2_de, spei_de \
        = data_exact(1982, 1999, 4, 10)


# %%
