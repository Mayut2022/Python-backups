# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 20:18:49 2022

@author: MaYutong
"""
# %%
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import os
import rasterio
import xarray as xr
from scipy.stats import linregress

lat = np.linspace(89.975, -89.975, 3600)
lon = np.linspace(-180, 179.95, 7200)

lat_spei = np.linspace(-89.75, 89.75, 360)
lon_spei = np.linspace(-179.75, 179.75, 720)

# %% 全球数据插值成0.5X0.5


def GLDAS_to_ESA(data):
    data = data*0.0001
    data[data > 3] = np.nan
    data = xr.DataArray(data, dims=['y', 'x'], coords=[lat, lon])
    data_ESA = data.interp(y=lat_spei, x=lon_spei, method="linear")

    return np.array(data_ESA)

# %%


def sif_xarray(band1):
    sif = xr.DataArray(band1, dims=['y', 'x'], coords=[lat, lon])
    return sif.loc[60:35, 100:150]


# %%
def SIF_all():
    inpath = (f'/mnt/e/Gosif_Monthly/data_01_20/')
    filelist = os.listdir(inpath)
    year = np.arange(2001, 2021, 1)

    for yr in year:
        sif_yr = []
        for i, file in enumerate(filelist):
            if str(yr) in file:
                print(file)

                inpath = (f'/mnt/e/Gosif_Monthly/data_01_20/{file}')
                ds = rasterio.open(inpath)
                sif = ds.read(1)

                # sif_RG = sif_xarray(sif)
                # lat_nc = np.array(sif_RG.y)
                # lon_nc = np.array(sif_RG.x)
                # sif_RG = np.array(sif_RG)
                sif_ESA = GLDAS_to_ESA(sif)
                sif_yr.append(sif_ESA)

        sif_yr = np.array(sif_yr)

        sif_yr = sif_yr.reshape(1, 12, 360, 720)

        if (yr == 2001):
            sif_all = sif_yr
        else:
            sif_all = np.vstack((sif_all, sif_yr))
        ds.close()

    # return sif_all, lat_nc, lon_nc
    return sif_all

# %%
def SIF_00():
    inpath = (f'/mnt/e/Gosif_Monthly/data_00/')
    filelist = os.listdir(inpath)

    
    sif_yr = []
    for i, file in enumerate(filelist):
        print(file)

        inpath = (f'/mnt/e/Gosif_Monthly/data_00/{file}')
        ds = rasterio.open(inpath)
        sif = ds.read(1)

        sif_ESA = GLDAS_to_ESA(sif)
        sif_yr.append(sif_ESA)

    sif_yr = np.array(sif_yr)

    return sif_yr

# %%


def trend(data):
    data = data*0.0001
    data[data == 3.2767] = np.nan
    data[data == 3.2766] = np.nan
    t = np.arange(1, 21, 1)
    s, r0, p = np.zeros((12, 500, 1001)), np.zeros(
        (12, 500, 1001)), np.zeros((12, 500, 1001))
    for mn in range(12):
        for r in range(500):
            if r % 25 == 0:
                print(f"{r} is done!")
            for c in range(1001):
                a = data[:, mn, r, c]
                if np.isnan(a).any():
                    s[mn, r, c], r0[mn, r, c], p[mn,
                                                 r, c] = np.nan, np.nan, np.nan
                else:
                    s[mn, r, c], _, r0[mn, r, c], p[mn,
                                                    r, c], _ = linregress(t, a)
        print(f"{mn} is done!")
    return s, p

# %%生成新的nc文件


def CreatNC(data):
    year = np.arange(2001, 2021, 1)
    month = np.arange(1, 13, 1)

    new_NC = nc.Dataset(
        r"/mnt/e/Gosif_Monthly/data_RG/GOSIF_01_20_RG.nc", 'w', format='NETCDF4')

    new_NC.createDimension('year', 20)
    new_NC.createDimension('month', 12)
    new_NC.createDimension('lat', 500)
    new_NC.createDimension('lon', 1001)

    var = new_NC.createVariable('sif', 'f', ("year", "month", "lat", "lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))

    new_NC.variables['sif'][:] = data
    new_NC.variables['lat'][:] = lat_nc
    new_NC.variables['lon'][:] = lon_nc

    var.sif = ("包含此前尝试的三个区域，region1 ESA IM")
    var.lat_range = "[60, 35], 500, 精度：0.05, 边界：[59.975, 35.025]"
    var.lon_range = "[100, 150], 1001, 精度：0.05, 边界：[100, 150]"
    var.data = "scale factor和缺测值均未处理"
    var.Fillvalues = "32767 (water bodies) and 32766 (lands under snow/ice throughout the year)"
    var.veg_nonveg = "annual mean <0 & 32767/32766"

    # 最后记得关闭文件
    new_NC.close()

# %%生成新的nc文件


def CreatNC2(data1, data2):
    month = np.arange(1, 13, 1)

    new_NC = nc.Dataset(
        r"/mnt/e/Gosif_Monthly/data_RG/GOSIF_01_20_RG_Trend.nc", 'w', format='NETCDF4')

    new_NC.createDimension('month', 12)
    new_NC.createDimension('lat', 500)
    new_NC.createDimension('lon', 1001)

    var = new_NC.createVariable('s', 'f', ("month", "lat", "lon"))
    new_NC.createVariable('p', 'f', ("month", "lat", "lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))

    new_NC.variables['s'][:] = data1
    new_NC.variables['p'][:] = data2
    new_NC.variables['lat'][:] = lat_nc
    new_NC.variables['lon'][:] = lon_nc

    var.sif = ("包含此前尝试的三个区域，region1 ESA IM")
    var.lat_range = "[60, 35], 500, 精度：0.05, 边界：[59.975, 35.025]"
    var.lon_range = "[100, 150], 1001, 精度：0.05, 边界：[100, 150]"
    var.data = "scale factor和缺测值均已处理，乘过0.0001"
    var.Fillvalues = "32767 (water bodies) and 32766 (lands under snow/ice throughout the year)"
    var.veg_nonveg = "annual mean <0 & 32767/32766"

    # 最后记得关闭文件
    new_NC.close()


# %%
# sif_all, lat_nc, lon_nc = SIF_all()
# s, p = trend(sif_all)
# CreatNC2(s, p)
# CreatNC(sif_all)

# %%

def CreatNC3(data):
    year = np.arange(2001, 2021, 1)
    month = np.arange(1, 13, 1)

    new_NC = nc.Dataset(
        r"/mnt/e/Gosif_Monthly/data_Global/GOSIF_01_20_SPEI0.5X0.5.nc", 'w', format='NETCDF4')

    new_NC.createDimension('year', 20)
    new_NC.createDimension('month', 12)
    new_NC.createDimension('lat', 360)
    new_NC.createDimension('lon', 720)

    var = new_NC.createVariable('sif', 'f', ("year", "month", "lat", "lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))

    new_NC.variables['sif'][:] = data
    new_NC.variables['lat'][:] = lat_spei
    new_NC.variables['lon'][:] = lon_spei

    var.data = "scale factor和缺测值均已处理, X0.01, >3 为缺测值"
    var.Fillvalues = "32767 (water bodies) and 32766 (lands under snow/ice throughout the year)"
    var.veg_nonveg = "annual mean <0 & 32767/32766"

    # 最后记得关闭文件
    new_NC.close()


# sif_all = SIF_all()
# CreatNC3(sif_all)

# %% 提取00-20的数据
# sif_00 = SIF_00()
# sif_all = SIF_all()
sif_00 = sif_00.reshape(1, 12, 360, 720)
sif_0020 = np.vstack((sif_00, sif_all))
CreatNC3(sif_0020)
# %%
