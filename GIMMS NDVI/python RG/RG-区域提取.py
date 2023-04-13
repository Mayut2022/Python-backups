# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 10:13:14 2022

@author: MaYutong
"""

import netCDF4 as nc
import numpy as np
import os
import xarray as xr
from scipy.stats import linregress

# %%


def read_nc(inpath):
    global lat, lon

    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        #print(f.variables['time'], "\n")
        # print(f.variables['ndvi'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        ndvi = (f.variables['ndvi'][:])
        time = (f.variables['time'][:])

    return ndvi

# %%


def ndvi_xarray(band1):
    t = np.arange(6)
    sif = xr.DataArray(band1, dims=['t', 'y', 'x'], coords=[t, lat, lon])
    return sif.loc[:, 60:35, 100:150]

# %% RG 插值


def GIMMS_to_ESA(data, lat, lon):
    # lat_spei = np.linspace(-89.75, 89.75, 360)
    # lon_spei = np.linspace(-179.75, 179.75, 720)
    # lat_spei = np.linspace(-59.75, 89.75, 300)
    data = data*0.0001
    data[data < -0.3] = np.nan
    lat_spei = np.linspace(35.25, 59.75, 50)
    lon_spei = np.linspace(100.25, 149.75, 100)
    t = np.arange(1, 35, 1)
    mn = np.arange(1, 13, 1)
    ndvi = xr.DataArray(data, dims=['t', 'mn', 'y', 'x'], coords=[
        t, mn, lat, lon])  # 原SPEI-base数据

    ndvi_ESA = ndvi.interp(t=t, mn=mn, y=lat_spei, x=lon_spei, method="linear")
    lat_ESA = ndvi_ESA.y
    lon_ESA = ndvi_ESA.x

    return ndvi_ESA, lat_ESA, lon_ESA

# %%


def ndvi_all():
    path = (r"E:/GIMMS_NDVI/data/")
    filelist = os.listdir(path)
    year = np.arange(1982, 2016, 1)

    for yr in year:
        ndvi_yr = []
        for i, file in enumerate(filelist):
            if str(yr) in file:
                print(file)
                inpath = (f'{path}/{file}')
                ndvi = read_nc(inpath)
                ndvi2 = ndvi.reshape(6, 2, 2160, 4320).mean(axis=1)

                ndvi_RG = ndvi_xarray(ndvi2)
                lat_nc = np.array(ndvi_RG.y)
                lon_nc = np.array(ndvi_RG.x)
                ndvi_RG = np.array(ndvi_RG)
                ndvi_yr.append(ndvi_RG)

        ndvi_yr = np.array(ndvi_yr).reshape(12, 300, 600)

        ndvi_yr = ndvi_yr.reshape(1, 12, 300, 600)

        if(yr == 1982):
            ndvi_all = ndvi_yr
        else:
            ndvi_all = np.vstack((ndvi_all, ndvi_yr))

    return ndvi_all, lat_nc, lon_nc

# %%


def trend(data):
    data = data*0.0001
    data[data < -0.3] = np.nan
    t = np.arange(1, 35, 1)
    s, r0, p = np.zeros((12, 300, 600)), np.zeros(
        (12, 300, 600)), np.zeros((12, 300, 600))
    for mn in range(12):
        for r in range(300):
            if r % 30 == 0:
                print(f"{r} is done!")
            for c in range(600):
                a = data[:, mn, r, c]
                if np.isnan(a).any():
                    s[mn, r, c], r0[mn, r, c], p[mn,
                                                 r, c] = np.nan, np.nan, np.nan
                else:
                    s[mn, r, c], _, r0[mn, r, c], p[mn,
                                                    r, c], _ = linregress(t, a)
        print(f"month {mn} is done! \n")
    return s, p


def trend2(data):
    data = data*0.0001
    data[data < -0.3] = np.nan
    t = np.arange(1, 35, 1)
    s, r0, p = np.zeros((300, 600)), np.zeros(
        (300, 600)), np.zeros((300, 600))
    
    for r in range(300):
        if r % 30 == 0:
            print(f"{r} is done!")
        for c in range(600):
            a = data[:, r, c]
            if np.isnan(a).any():
                s[r, c], r0[r, c], p[r, c] = np.nan, np.nan, np.nan
                                             
            else:
                s[r, c], _, r0[r, c], p[r, c], _ = linregress(t, a)
                                                
    return s, p

# %%生成新的nc文件


def CreatNC(data):
    year = np.arange(1982, 2016, 1)
    month = np.arange(1, 13, 1)

    new_NC = nc.Dataset(
        r"E:/GIMMS_NDVI/data_RG/NDVI_82_15_RG.nc", 'w', format='NETCDF4')

    new_NC.createDimension('year', 34)
    new_NC.createDimension('month', 12)
    new_NC.createDimension('lat', 300)
    new_NC.createDimension('lon', 600)

    var = new_NC.createVariable('ndvi', 'f', ("year", "month", "lat", "lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))

    new_NC.variables['ndvi'][:] = data
    new_NC.variables['lat'][:] = lat_nc
    new_NC.variables['lon'][:] = lon_nc

    var.ndvi = ("包含此前尝试的三个区域，region1 ESA IM")
    var.data = "scale factor和缺测值均未处理"
    var.Fillvalues = "-32767; scaleX10000; missingvalue:-5000; valid_range:[-0.3, 1]"
    var.veg_nonveg = "annual mean <0 & 32767/32766"

    # 最后记得关闭文件
    new_NC.close()

# %%生成新的nc文件


def CreatNC2(data1, data2):
    month = np.arange(1, 13, 1)

    new_NC = nc.Dataset(
        r"E:/GIMMS_NDVI/data_RG/NDVI_82_15_RG_Trend.nc", 'w', format='NETCDF4')

    new_NC.createDimension('month', 12)
    new_NC.createDimension('lat', 300)
    new_NC.createDimension('lon', 600)

    var = new_NC.createVariable('s', 'f', ("month", "lat", "lon"))
    new_NC.createVariable('p', 'f', ("month", "lat", "lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))

    new_NC.variables['s'][:] = data1
    new_NC.variables['p'][:] = data2
    new_NC.variables['lat'][:] = lat_nc
    new_NC.variables['lon'][:] = lon_nc

    var.ndvi = ("包含此前尝试的三个区域，region1 ESA IM")
    var.data = "scale factor和缺测值均未处理"
    var.Fillvalues = "-32767; scaleX10000; missingvalue:-5000; valid_range:[-0.3, 1]"
    var.veg_nonveg = "annual mean <0 & 32767/32766"
    var.data = "scale factor和缺测值均已处理，乘过0.0001"
    var.veg_nonveg = "annual mean <0 & 32767/32766"

    # 最后记得关闭文件
    new_NC.close()

# %%生成新的nc文件


def CreatNC3(data):
    year = np.arange(1982, 2016, 1)
    month = np.arange(1, 13, 1)

    new_NC = nc.Dataset(
        r"E:/GIMMS_NDVI/data_RG/NDVI_82_15_RG_SPEI0.5x0.5.nc", 'w', format='NETCDF4')

    new_NC.createDimension('year', 34)
    new_NC.createDimension('month', 12)
    new_NC.createDimension('lat', 50)
    new_NC.createDimension('lon', 100)

    var = new_NC.createVariable('ndvi', 'f', ("year", "month", "lat", "lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))

    new_NC.variables['ndvi'][:] = data
    new_NC.variables['lat'][:] = lat_RG
    new_NC.variables['lon'][:] = lon_RG

    var.ndvi = ("包含此前尝试的三个区域，region1 ESA IM")
    var.data = "scale factor和缺测值均已处理，<-0.3为np.nan"
    var.Fillvalues = "-32767; scaleX10000; missingvalue:-5000; valid_range:[-0.3, 1]"
    var.veg_nonveg = "annual mean <0 & 32767/32766"

    # 最后记得关闭文件
    new_NC.close()


# %% 生长季趋势

def CreatNC4(data1, data2):
    new_NC = nc.Dataset(
        r"E:/GIMMS_NDVI/data_RG/NDVI_82_15_RG_GSL_Trend.nc", 'w', format='NETCDF4')

    new_NC.createDimension('lat', 300)
    new_NC.createDimension('lon', 600)

    var = new_NC.createVariable('s', 'f', ("lat", "lon"))
    new_NC.createVariable('p', 'f', ("lat", "lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))

    new_NC.variables['s'][:] = data1
    new_NC.variables['p'][:] = data2
    new_NC.variables['lat'][:] = lat_nc
    new_NC.variables['lon'][:] = lon_nc

    var.ndvi = ("包含此前尝试的三个区域，region1 ESA IM")
    var.data = "scale factor和缺测值均未处理"
    var.Fillvalues = "-32767; scaleX10000; missingvalue:-5000; valid_range:[-0.3, 1]"
    var.veg_nonveg = "annual mean <0 & 32767/32766"
    var.data = "scale factor和缺测值均已处理，乘过0.0001"
    var.veg_nonveg = "annual mean <0 & 32767/32766"

    # 最后记得关闭文件
    new_NC.close()


# %%
ndvi_all, lat_nc, lon_nc = ndvi_all()
# ndvi_RG, lat_RG, lon_RG = GIMMS_to_ESA(ndvi_all, lat_nc, lon_nc)
# CreatNC3(ndvi_RG)
s, p = trend(ndvi_all)
# CreatNC2(s, p)
ndvi_gsl = np.nanmean(ndvi_all[:, 3:9, :, :], axis=1)
s2, p2 = trend2(ndvi_gsl)
CreatNC4(s2, p2)
# %%
'''
import matplotlib.pyplot as plt

for i in range(12):
    plt.figure(3, dpi=500)
    plt.imshow(ndvi_RG[0, i, :, :], cmap='Set3')
    plt.colorbar(shrink=0.75)
    plt.show()
'''
