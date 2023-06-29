# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 14:22:02 2022

@author: MaYutong
"""

import netCDF4 as nc
import cftime
import xarray as xr
import numpy as np
from scipy import signal

# %%
inpath = (r'E:/SPEI_base/data/spei03.nc')
with nc.Dataset(inpath) as f:
    print(f.variables.keys())
    print(f.variables['spei'])

    spei = (f.variables['spei'][:])
    lat = (f.variables['lat'][:])
    lon = (f.variables['lon'][:])

    # print(f.variables['time'])
    time = (f.variables['time'][:])
    t = nc.num2date(f.variables['time'][:],
                    'days since 1900-01-01 00:00:0.0').data

# %%


def region1(data):
    spei_global = xr.DataArray(data, dims=['t', 'y', 'x'], coords=[
                               time, lat, lon])  # 原SPEI-base数据
    spei_IM = spei_global.loc[:, 40:55, 100:125]

    return spei_IM


spei_IM = region1(spei)
lat_IM = spei_IM.y
lon_IM = spei_IM.x

# %%

a = 0
b = 12
for i in range(120):
    c = np.nanmean(spei_IM[a:b, :, :], axis=0)
    c = c.reshape(1, 30, 50)
    if a == 0:
        spei_IM_ave = c
        a = a+12
        b = b+12
    else:
        spei_IM_ave = np.vstack((spei_IM_ave, c))
        a = a+12
        b = b+12

del a, b, c


# %%生成新的nc文件
def CreatNC(data):
    new_NC = nc.Dataset(
        r"E:/SPEI_base/data/spei03_MG_annual.nc", 'w', format='NETCDF4')

    time = np.arange(1901, 2021, 1)
    new_NC.createDimension('time', 120)
    new_NC.createDimension('lat', 30)
    new_NC.createDimension('lon', 50)

    var = new_NC.createVariable('spei', 'f', ("time", "lat", "lon"))
    new_NC.createVariable('time', 'f', ("time"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))

    new_NC.variables['spei'][:] = data
    new_NC.variables['time'][:] = time
    new_NC.variables['lat'][:] = lat_IM
    new_NC.variables['lon'][:] = lon_IM

    #var.lat_range="[-37, 35], 144, 精度：0.5, 边界：[-34.75, 36.75]"
    #var.lon_range="[-18, 52], 140, 精度：0.5, 边界：[-17.75, 51.75]"
    var.Fillvalues = "nan"
    var.time = "用SPEI03算出来的年平均，1-12月SPEI03指数"

    # 最后记得关闭文件
    new_NC.close()

# CreatNC(spei_IM_ave)

# %%生成新的nc文件


def CreatNC2(data):
    new_NC = nc.Dataset(r"E:/SPEI_base/data/spei03_MG.nc",
                        'w', format='NETCDF4')

    new_NC.createDimension('time', 1440)
    new_NC.createDimension('lat', 30)
    new_NC.createDimension('lon', 50)

    var = new_NC.createVariable('spei', 'f', ("time", "lat", "lon"))
    new_NC.createVariable('time', 'f', ("time"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))

    new_NC.variables['spei'][:] = data
    new_NC.variables['time'][:] = time
    new_NC.variables['lat'][:] = lat_IM
    new_NC.variables['lon'][:] = lon_IM

    var.Fillvalues = "nan"
    var.time = "1901.1-2020.12 'days since 1900-01-01 00:00:0.0' "

    # 最后记得关闭文件
    new_NC.close()

# CreatNC2(spei_IM)

# %%


def mn_yr(data):
    spei_mn = []
    for mn in range(12):
        spei_ = []
        for yr in range(120):
            spei_.append(data[mn])
            mn += 12
        spei_mn.append(spei_)

    spei_mn = np.array(spei_mn)

    return spei_mn


def season_yr(data):
    spei_s = np.vstack((data[2:, :], data[:2, :]))
    spei_sea = []
    for mn1, mn2 in zip(range(0, 12, 3), range(3, 15, 3)):
        spei_sea.append(spei_s[mn1:mn2, :])

    spei_sea = np.array(spei_sea)
    spei_sea = spei_sea.mean(axis=1)

    return spei_sea


spei_mn = mn_yr(spei_IM)
spei_sea = season_yr(spei_mn)

# %%


def CreatNC3(data):
    new_NC = nc.Dataset(
        r"E:/SPEI_base/data/spei03_MG_season.nc", 'w', format='NETCDF4')

    new_NC.createDimension('season', 4)
    new_NC.createDimension('time', 120)
    new_NC.createDimension('lat', 30)
    new_NC.createDimension('lon', 50)

    var = new_NC.createVariable('spei', 'f', ("season", "time", "lat", "lon"))
    new_NC.createVariable('time', 'f', ("time"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))

    new_NC.variables['spei'][:] = data
    new_NC.variables['lat'][:] = lat_IM
    new_NC.variables['lon'][:] = lon_IM

    var.Fillvalues = "nan"
    var.time = "1901.1-2020.12 'days since 1900-01-01 00:00:0.0' "

    # 最后记得关闭文件
    new_NC.close()

# CreatNC3(spei_sea)

# %%


def CreatNC4(data):
    new_NC = nc.Dataset(r"E:/SPEI_base/data/spei03_MG_detrend.nc",
                        'w', format='NETCDF4')
    
    month = np.arange(12)
    year = np.arange(40)
    
    new_NC.createDimension('mn', 12)
    new_NC.createDimension('yr', 40)
    new_NC.createDimension('lat', 30)
    new_NC.createDimension('lon', 50)

    var = new_NC.createVariable('spei', 'f', ("mn", "yr", "lat", "lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))

    new_NC.variables['spei'][:] = data
    new_NC.variables['lat'][:] = lat_IM
    new_NC.variables['lon'][:] = lon_IM

    var.Fillvalues = "nan"
    var.time = "1981.1-2020.12"

    # 最后记得关闭文件
    new_NC.close()


# %% 数据去趋势 1981-2020


def detrend(data):
    data_detrend = data.copy()
    for mn in range(12):
        for r in range(30):
            for c in range(50):
                test = data_detrend[mn, :, r, c]
                contain_nan = (True in np.isnan(test))
                if contain_nan == True:
                    continue
                else:
                    detrend = signal.detrend(test, type='linear')
                    data_detrend[mn, :, r, c] = detrend+test.mean()

    return data_detrend


spei_mn2 = spei_mn[:, 80:, :]
spei_mn2_de = detrend(spei_mn2)
CreatNC4(spei_mn2_de)
