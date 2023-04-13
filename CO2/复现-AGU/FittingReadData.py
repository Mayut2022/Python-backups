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
lat_RG = np.linspace(35.25, 59.75, 50)
lon_RG = np.linspace(100.25, 149.75, 100)
lat_g = np.linspace(-89.75, 89.75, 360)
lon_g = np.linspace(-179.75, 179.75, 720)


# %% 读取因变量Y


class Y():
    name = "1ndvi; 2gpp"

    def read(self, inpath, var):
        # print(inpath)
        with nc.Dataset(inpath) as f:
            # print(f.variables.keys())
            # lat = (f.variables['lat'][:])
            # lon = (f.variables['lon'][:])
            data = (f.variables[f'{var}'][:]).data
        return data


# X为Y的子类，继承Y


class X(Y):
    name = "1TMP; 2Pre; 3CO2"

# %% 通用提取


class XR():
    description = "3 dimensions, (mn, lat, lon)"

    def rg(self, data):
        mn = np.arange(data.shape[0])
        data_xr = xr.DataArray(data, dims=['mn', 'y', 'x'], coords=[
            mn, lat_RG, lon_RG])

        data_xr_MG = data_xr.loc[:, 40:55, 100:125]
        return np.array(data_xr_MG)

    def glob(self, data):
        mn = np.arange(data.shape[0])
        data_xr = xr.DataArray(data, dims=['mn', 'y', 'x'], coords=[
            mn, lat_g, lon_g])

        data_xr_MG = data_xr.loc[:, 40:55, 100:125]
        return np.array(data_xr_MG)

    # ndvi
    def yr_rg(self, data):
        yr = np.arange(data.shape[0])
        mn = np.arange(data.shape[1])
        data_xr = xr.DataArray(data, dims=['yr', 'mn', 'y', 'x'], coords=[
            yr, mn, lat_RG, lon_RG])

        data_xr_MG = data_xr.loc[:, :, 40:55, 100:125]
        return np.array(data_xr_MG)

    # vpd
    def yr_glob(self, data):
        yr = np.arange(data.shape[1])
        mn = np.arange(data.shape[0])
        data_xr = xr.DataArray(data, dims=['mn', 'yr', 'y', 'x'], coords=[
            mn, yr, lat_g, lon_g])

        data_xr_MG = data_xr.loc[:, :, 40:55, 100:125]
        return np.array(data_xr_MG)

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
# Rd: 读取+xr裁剪+mn_yr
# Rd: pre, co2

class Rd:
    y = Y()
    x = X()
    xra = XR()
    mnyr = MnYr()

    def read(self, inpath, var):
        data1 = self.x.read(inpath, var)

        var_glob = ["pre", "tmp"]
        var_rg = ["co2", "sr"]
        if var in var_glob:
            data2 = self.xra.glob(data1)
        elif var in var_rg:
            data2 = self.xra.rg(data1)

        data3 = self.mnyr.mn_yr(data2)

        return data3


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

    inpath_ndvi = r"/mnt/e/GIMMS_NDVI/data_RG/NDVI_82_15_RG_SPEI0.5x0.5.nc"
    ndvi = Y().read(inpath_ndvi, "ndvi")
    ndvi = XR().yr_rg(ndvi)

    gpp = y_GPP()

    #################

    inpath_pre = r"/mnt/e/CRU/Q_DATA_CRU-GLEAM/PRE_global_81_20.nc"
    inpath_co2_D = r"/mnt/e/CO2/DENG/CO2_81_13_RG_SPEI0.5x0.5.nc"
    inpath_tmp = r"/mnt/e/CRU/TMP_DATA/TMP_CRU_ORIGINAL_81_20.nc"
    inpath_vpd = r"/mnt/e/CRU/VAP_DATA/vpd_CRU_MONTH_81_20.nc"
    inpath_sr = r"/mnt/e/GLDAS Noah/DATA_RG/sr_81_20_ORINGINAL_SPEI0.5x0.5.nc"
    inpath_co2_CT = r"/mnt/e/CO2/CT2019B/molefractions/XCO2 Global/XCO2_00_18_SPEI0.5x0.5.nc"
    pre = rd.read(inpath_pre, "pre")
    co2 = rd.read(inpath_co2_D, "co2")
    tmp = rd.read(inpath_tmp, "tmp")
    vpd = X().read(inpath_vpd, "vpd")
    vpd = XR().yr_glob(vpd)
    sr = rd.read(inpath_sr, "sr")
    co2_CT = X().read(inpath_co2_CT, "co2")
    co2_CT = XR().yr_glob(co2_CT)

    print("ndvi, gpp, co2, pre, tmp, vpd, sr, co2_CT")
    return ndvi, gpp, co2, pre, tmp, vpd, sr, co2_CT


ndvi, gpp, co2, pre, tmp, vpd, sr, co2_CT = data_read()

# %%


class Data_exact:
    input = "input the year like: 1981, 1982, etc. "

    def exact0(self, data, sy, ey, sm, em):
        print("ndvi, from 1982")
        data = data[sy-1982:ey-1982+1, sm-1:em, :, :]
        return data

    def exact1(self, data, sy, ey, sm, em):
        print("gpp, from 1982")
        data = data[sy-1982:ey-1982+1, sm-1:em, :, :]
        return data

    def exact2(self, data, sy, ey, sm, em):
        print("co2 Deng 1981-2013")
        data = data[sm-1:em, sy-1981:ey-1981+1, :, :]
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
    ndvi_de = de.exact0(ndvi, sy, ey, sm, em)
    gpp_de = de.exact1(gpp, sy, ey, sm, em)
    co2_de = de.exact2(co2, sy, ey, sm, em)
    co2_ct_de = de.exact4(co2_CT, sy, ey, sm, em)

    
    var = ["pre", "tmp", "vpd", "sr"]
    for x in var:
        exec(f"{x}_de = de.exact3({x}, sy, ey, sm, em)", globals())

    print("ndvi, gpp, co2, pre, tmp, vpd, sr, co2_CT")
    return ndvi_de, gpp_de, co2_de, pre_de, tmp_de, vpd_de, sr_de, co2_ct_de

def data_time(a, b, c, d):
    global sy, ey, sm, em
    sy, ey, sm, em = a, b, c, d
    
# %% 测试
if __name__ == "__main__":
    pass
    # rd = Rd()

    # inpath_ndvi = r"/mnt/e/GIMMS_NDVI/data_RG/NDVI_82_15_RG_SPEI0.5x0.5.nc"
    # ndvi = Y().read(inpath_ndvi, "ndvi")
    # ndvi = XR().yr_rg(ndvi)

    # gpp = y_GPP()

    # #################

    # inpath_pre = r"/mnt/e/CRU/Q_DATA_CRU-GLEAM/PRE_global_81_20.nc"
    # inpath_co2_D = r"/mnt/e/CO2/DENG/CO2_81_13_RG_SPEI0.5x0.5.nc"
    # inpath_tmp = r"/mnt/e/CRU/TMP_DATA/TMP_CRU_ORIGINAL_81_20.nc"
    # inpath_vpd = r"/mnt/e/CRU/VAP_DATA/vpd_CRU_MONTH_81_20.nc"
    # inpath_sr = r"/mnt/e/GLDAS Noah/DATA_RG/sr_81_20_ORINGINAL_SPEI0.5x0.5.nc"
    # pre = rd.read(inpath_pre, "pre")
    # co2 = rd.read(inpath_co2_D, "co2")
    # tmp = rd.read(inpath_tmp, "tmp")
    # vpd = X().read(inpath_vpd, "vpd")
    # vpd = XR().yr_glob(vpd)
    # sr = rd.read(inpath_sr, "sr")

#%%
if __name__ == "__main__":
    ##### 读取全部数据
    ndvi, gpp, co2, pre, tmp, vpd, sr, co2_CT = data_read()

    data_time(2000, 2018, 4, 10)
    ndvi_de, gpp_de, *x_data = data_exact(2000, 2018, 4, 10)

    x_var = ["co2", "pre", "tmp", "vpd", "sr", "co2_CT"]
    for x, xd in zip(x_var, x_data):
        exec(f"{x}_de = xd")
        del x, xd
# %%
