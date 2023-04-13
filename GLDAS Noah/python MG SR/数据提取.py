# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 16:37:59 2022

@author: MaYutong
"""
# %%
import cftime
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import os
import pandas as pd
# from sklearn import preprocessing
import xarray as xr

# %%


def read_nc1(inpath, mn):
    global lat, lon
    with nc.Dataset(inpath) as f:
        '''
        print(f.variables.keys())
        print(f.variables['SWdown_f_tavg']) # units: W m-2 ; month mean
        print(f.variables['time'])
        print("")
        '''
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        # lcc = (f.variables['lcc'][:])
        time = (f.variables['time'][:])
        t = nc.num2date(time, 'days since 1948-01-01 00:00:0.0').data
        sr = (f.variables['SWdown_f_tavg'][:])

        return sr


# %%
def sr_1_all():
    year = np.arange(1981, 2001, 1)
    month = np.arange(1, 13, 1)
    i = 0
    for yr in year:
        for mn in month:
            date = str(yr)+str(mn).zfill(2)
            inpath = (
                rf"E:/GLDAS Noah/DATA_GLOBAL/SR_V2.0_81_00/GLDAS_NOAH025_M.A{date}.020.nc4.SUB.nc4")
            print(inpath)
            sr_1 = read_nc1(inpath, mn)
            if yr == 1981 and mn == 1:
                sr = sr_1
            else:
                sr = np.vstack((sr, sr_1))

    return sr


def sr_2_all():
    year = np.arange(2001, 2021, 1)
    month = np.arange(1, 13, 1)
    i = 0
    for yr in year:
        for mn in month:
            date = str(yr)+str(mn).zfill(2)
            inpath = (
                rf"E:/GLDAS Noah/DATA_GLOBAL/SR_V2.1_01_20/GLDAS_NOAH025_M.A{date}.021.nc4.SUB.nc4")
            print(inpath)
            sr_1 = read_nc1(inpath, mn)
            if yr == 2001 and mn == 1:
                sr = sr_1
            else:
                sr = np.vstack((sr, sr_1))

    return sr


sr1 = sr_1_all()
sr2 = sr_2_all()
sr = np.vstack((sr1, sr2))
sr = sr.data

# %%


def CreatNC():
    new_NC = nc.Dataset(
        rf"E:/GLDAS Noah/DATA_GLOBAL/SR_81_20_global/sr_ORINGINAL.nc",
        'w', format='NETCDF4')

    time = pd.date_range("1981-01-01", periods=480, freq="MS")

    new_NC.createDimension('time', 480)
    new_NC.createDimension('lat', len(lat))
    new_NC.createDimension('lon', len(lon))

    var = new_NC.createVariable('sr', 'f', ("time", "lat", "lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))

    new_NC.variables['sr'][:] = sr
    new_NC.variables['lat'][:] = lat
    new_NC.variables['lon'][:] = lon

    var.description = "1981.1-2020.12 sr 仅合并未处理原始数据 monthly mean"
    var.time = 'time = pd.date_range("1981-01-01", periods=480, freq="MS")'
    var.unit = "units: W m-2"
    var.missing_value = "-9999.0"

    new_NC.close()


# CreatNC()

# %% 提取区域RG
def region(data):
    t = pd.date_range("1981-01-01", periods=480, freq="MS")
    sr_global = xr.DataArray(data, dims=['t', 'y', 'x'], coords=[
        t, lat, lon])  # 原SPEI-base数据
    sr_rg = sr_global.loc[:, 35:60, 100:150]

    lat_rg = sr_rg.y
    lon_rg = sr_rg.x

    return sr_rg, lat_rg, lon_rg


# sr_rg, lat_rg, lon_rg = region(sr)

# %%
def CreatNC2(data):
    new_NC = nc.Dataset(
        rf"E:/GLDAS Noah/DATA_RG/sr_81_20_ORINGINAL.nc",
        'w', format='NETCDF4')

    new_NC.createDimension('time', 480)
    new_NC.createDimension('lat', len(lat_rg))
    new_NC.createDimension('lon', len(lon_rg))

    var = new_NC.createVariable('sr', 'f', ("time", "lat", "lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))

    new_NC.variables['sr'][:] = data
    new_NC.variables['lat'][:] = lat_rg
    new_NC.variables['lon'][:] = lon_rg

    var.description = "1981.1-2020.12 sr 仅合并未处理原始数据 monthly mean"
    var.time = 'time = pd.date_range("1981-01-01", periods=480, freq="MS")'
    var.unit = "units: W m-2"
    var.missing_value = "-9999.0"

    new_NC.close()

# CreatNC2(sr_rg)

# %% RG 插值


def GLDAS_to_ESA(data, lat, lon):
    # lat_spei = np.linspace(-89.75, 89.75, 360)
    # lon_spei = np.linspace(-179.75, 179.75, 720)
    # lat_spei = np.linspace(-59.75, 89.75, 300)
    data = np.array(data)
    data[data == -9999] = np.nan
    lat_spei = np.linspace(35.25, 59.75, 50)
    lon_spei = np.linspace(100.25, 149.75, 100)
    t = pd.date_range("1981-01-01", periods=480, freq="MS")
    sr = xr.DataArray(data, dims=['t', 'y', 'x'], coords=[
        t, lat, lon])  # 原SPEI-base数据

    sr_ESA = sr.interp(t=t, y=lat_spei, x=lon_spei, method="linear")
    lat_ESA = sr_ESA.y
    lon_ESA = sr_ESA.x

    return sr_ESA, lat_ESA, lon_ESA

# sr_ESA, lat_ESA, lon_ESA = GLDAS_to_ESA(sr_rg, lat_rg, lon_rg)


def CreatNC3(data, lat, lon):
    new_NC = nc.Dataset(
        rf"E:/GLDAS Noah/DATA_RG/sr_81_20_ORINGINAL_SPEI0.5x0.5.nc",
        'w', format='NETCDF4')

    new_NC.createDimension('time', 480)
    new_NC.createDimension('lat', len(lat))
    new_NC.createDimension('lon', len(lon))

    var = new_NC.createVariable('sr', 'f', ("time", "lat", "lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))

    new_NC.variables['sr'][:] = data
    new_NC.variables['lat'][:] = lat
    new_NC.variables['lon'][:] = lon

    var.description = "1981.1-2020.12 sr 插值成0.5x0.5,未处理其他 monthly mean"
    var.time = 'time = pd.date_range("1981-01-01", periods=480, freq="Y")'
    var.unit = "units: W m-2"
    var.missingvalue = "-9999.0 计算过程中在插值前已处理为np.nan"

    new_NC.close()

# CreatNC3(sr_ESA, lat_ESA, lon_ESA)

# %% RG插值后标准化和异常值


def CreatNC4(data1, data2, mn, lat, lon):
    new_NC = nc.Dataset(
        rf"E:/GLDAS Noah/DATA_RG/Zscore_Anomaly/sr_81_20_SPEI0.5x0.5_Zscore_Anomaly_month{mn}.nc",
        'w', format='NETCDF4')

    year = np.arange(1, 41, 1)

    new_NC.createDimension('year', 40)
    new_NC.createDimension('layer', 4)
    new_NC.createDimension('lat', len(lat))
    new_NC.createDimension('lon', len(lon))

    var = new_NC.createVariable('sr_z', 'f', ("year", "layer", "lat", "lon"))
    new_NC.createVariable('sr_a', 'f', ("year", "layer", "lat", "lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))

    new_NC.variables['sr_z'][:] = data1
    new_NC.variables['sr_a'][:] = data2
    new_NC.variables['lat'][:] = lat
    new_NC.variables['lon'][:] = lon

    var.description = "1981.1-2020.12 sr 插值成0.5x0.5,未处理其他 monthly mean"
    var.time = 'time = pd.date_range("1981-01-01", periods=480, freq="Y")'
    var.unit = "kg m-2, 1 kg m-2 = 1 mm"
    var.missingvalue = "-9999.0，计算过程中已处理为np.nan"

    # 最后记得关闭文件
    new_NC.close()

# %% 480(输入data) -> 月 年


def mn_yr(data):
    sr_mn = []
    for mn in range(12):
        sr_ = []
        for yr in range(40):
            sr_.append(data[mn])
            mn += 12
        sr_mn.append(sr_)

    sr_mn = np.array(sr_mn)

    return sr_mn

# sr_mn = mn_yr(sr_ESA)


def Zscore(data):
    sr_mn_z = np.zeros((40, 4, 50, 100))
    sr_mn_a = np.zeros((40, 4, 50, 100))
    for mn in range(12):
        for l in range(4):
            for r in range(50):
                if r % 25 == 0:
                    print(f"columns {r} is done!")
                for c in range(100):
                    sr_mn_z[:, l, r, c] = preprocessing.scale(
                        data[mn, :, l, r, c])
                    ave = np.nanmean(sr_mn_z[:, l, r, c], axis=0)
                    for yr in range(40):
                        sr_mn_a[yr, l, r, c] = data[mn, yr, l, r, c] - ave

        CreatNC4(sr_mn_z, sr_mn_a, mn+1, lat_ESA, lon_ESA)

# Zscore(sr_mn)


# %% 全球数据插值成0.5X0.5
def GLDAS_to_ESA(data):
    data[data == -9999] = np.nan
    lat_spei = np.linspace(-89.75, 89.75, 360)
    lon_spei = np.linspace(-179.75, 179.75, 720)
    # lat_spei = np.linspace(-59.75, 89.75, 300)
    t = pd.date_range("1981-01-01", periods=480, freq="MS")
    sr = xr.DataArray(data, dims=['t', 'y', 'x'], coords=[
        t, lat, lon])  # 原SPEI-base数据

    sr_ESA = sr.interp(t=t, y=lat_spei, x=lon_spei, method="linear")
    lat_ESA = sr_ESA.y
    lon_ESA = sr_ESA.x

    return np.array(sr_ESA), lat_ESA, lon_ESA


sr_ESA, lat_ESA, lon_ESA = GLDAS_to_ESA(sr)


# %%
def CreatNC5(data):
    new_NC = nc.Dataset(
        rf"E:/GLDAS Noah/DATA_GLOBAL/SR_81_20_global/sr_ORINGINAL_SPEI_0.5X0.52.nc",
        'w', format='NETCDF4')

    time = pd.date_range("1981-01-01", periods=480, freq="MS")

    new_NC.createDimension('time', 480)
    new_NC.createDimension('lat', len(lat_ESA))
    new_NC.createDimension('lon', len(lon_ESA))

    var = new_NC.createVariable('sr', 'f', ("time",  "lat", "lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))

    new_NC.variables['sr'][:] = data
    new_NC.variables['lat'][:] = lat_ESA
    new_NC.variables['lon'][:] = lon_ESA

    var.description = "1981.1-2020.12 sr 仅合并未处理原始数据 monthly mean"
    var.time = 'time = pd.date_range("1981-01-01", periods=480, freq="MS")'
    var.missingvalue = "-9999.0 已处理过为np.nan"
    var.unit = "units: W m-2"

    new_NC.close()


# CreatNC5(sr_ESA)



# %%
plt.figure(1, dpi=300)
plt.imshow(sr_ESA[5, :, :], cmap='jet')  # 显示图像
plt.colorbar()
plt.savefig("./111.jpg")# plt.show()

