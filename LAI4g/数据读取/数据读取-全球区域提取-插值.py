# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 10:43:55 2023

@author: MaYutong
"""

import netCDF4 as nc
import numpy as np
import os
import rasterio
from sklearn import preprocessing
import xarray as xr

# %%
_ = np.load("latlon.npz")
lat, lon = _["lat"], _["lon"]


def lai_xarray(band1):
    data = xr.DataArray(band1, dims=['y', 'x'], coords=[lat, lon])
    data_MG = data.loc[55:40, 100:125]
    return np.array(data_MG), data_MG.y, data_MG.x


def read_tif(inpath):
    # global lat_MG, lon_MG
    with rasterio.open(inpath) as ds:
        band1 = ds.read(1)
    # data, lat_MG, lon_MG = lai_xarray(band1)
    data = band1.copy()
    data = data*0.01
    data[data == 655.35] = np.nan

    return data


# %%生成新的nc文件


def CreatNC(data, yr_len):
    sp = data.shape[0]
    year = np.arange(sp)
    month = np.arange(1, 13, 1)

    new_NC = nc.Dataset(
        rf"E:/LAI4g/data_Global/LAI_Global_{yr_len}.nc", 'w', format='NETCDF4')

    new_NC.createDimension('year', sp)
    new_NC.createDimension('month', 12)
    new_NC.createDimension('lat', 2160)
    new_NC.createDimension('lon', 4320)

    var = new_NC.createVariable('lai', 'f', ("year", "month", "lat", "lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))

    new_NC.variables['lai'][:] = data
    new_NC.variables['lat'][:] = lat
    new_NC.variables['lon'][:] = lon

    var.data = "scale factor和缺测值均已处理，65535为np.nan"
    var.units = "m2/m2 (month mean)"

    # 最后记得关闭文件
    new_NC.close()
# %%


def lai_all(path):
    a = path.split('/')
    b = a[2].split('_')
    ys, ye = b[0], b[1]
    year = np.arange(int(ys), int(ye)+1)
    month = []
    [month.append(str(x).zfill(2)) for x in range(1, 13)]

    filename = os.listdir(path)

    lai_yr = []
    for yr in year[:]:
        lai_mn = []
        for mn in month:
            lai_ = []
            date = str(yr)+mn
            for file in filename:
                if date in file:
                    inpath = path+file
                    print(inpath)
                    lai = read_tif(inpath)
                    lai_.append(lai)  # 每个月两天
            lai_ = np.array(lai_)
            lai_ = np.nanmean(lai_, axis=0)
            lai_mn.append(lai_)  # 每年数据
        lai_yr.append(lai_mn)
    lai_yr = np.array(lai_yr)

    return lai_yr


path1 = r"E:/LAI4g/1982_1990_TIFF/"
path2 = r"E:/LAI4g/1991_2000_TIFF/"
path3 = r"E:/LAI4g/2001_2010_TIFF/"
path4 = r"E:/LAI4g/2011_2020_TIFF/"

yr_str = ["1982_1990", "1991_2000", "2001_2010", "2011_2020"]

# for i in range(2, 5):
#     data = eval(f"lai_all(path{i})")
#     CreatNC(data, yr_str[i-1])

'''
数据有问题，1994年7月两数据都没有，因此循环会出问题
200605和200908两个月数据分别缺一个，则程序运行无问题
list append 函数：(180, 300)->(1, 180, 300)
'''


# %% 数据插值成SPEI精度


def GIMMS_to_SPEI(data, lat, lon):
    global lat_spei, lon_spei
    lat_spei = np.linspace(89.75, -89.75, 360)
    lon_spei = np.linspace(-179.75, 179.75, 720)

    # sp = data.shape[0]
    # t = np.arange(sp)
    mn = np.arange(1, 13, 1)
    ndvi = xr.DataArray(data, dims=['mn', 'y', 'x'], coords=[
        mn, lat, lon])  # 原SPEI-base数据

    ndvi_SPEI = ndvi.interp(mn=mn, y=lat_spei,
                            x=lon_spei, method="linear")
    lat_SPEI = ndvi_SPEI.y
    lon_SPEI = ndvi_SPEI.x

    return ndvi_SPEI, lat_SPEI, lon_SPEI


# %%


def CreatNC2(data, yr):
    month = np.arange(1, 13, 1)

    new_NC = nc.Dataset(
        rf"E:/LAI4g/data_Global/0.5X0.5/LAI_Global_{yr}_0.5x0.5.nc", 'w', format='NETCDF4')

    
    new_NC.createDimension('month', 12)
    new_NC.createDimension('lat', 360)
    new_NC.createDimension('lon', 720)

    var = new_NC.createVariable('lai', 'f', ("month", "lat", "lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))

    new_NC.variables['lai'][:] = data
    new_NC.variables['lat'][:] = lat_spei
    new_NC.variables['lon'][:] = lon_spei

    var.data = "scale factor和缺测值均已处理，65535为np.nan"
    var.units = "m2/m2 (month mean)"
    var.description = "注意lat顺序，原顺序为由北向南，现顺序为由南向北"

    # 最后记得关闭文件
    new_NC.close()


# %%
yr_all = [np.arange(1982, 1991), np.arange(1991, 2001), np.arange(2001, 2011), np.arange(2011, 2021)]
for i in range(3, 5):
    data = eval(f"lai_all(path{i})")
    for j, yr in enumerate(yr_all[i-1]):
        data_SPEI, _, _ = GIMMS_to_SPEI(data[j], lat, lon)
        CreatNC2(data_SPEI, yr)


# %% Z-score归一化值

def Zscore(data):
    data_z = np.zeros((39, 12, 30, 50))
    for mn in range(12):
        for r in range(30):
            if r % 10 == 0:
                print(f"columns {r} is done!")
            for c in range(50):
                data_z[:, mn, r, c] = preprocessing.scale(data[:, mn, r, c])
    return data_z


# lai_SPEI_z = Zscore(lai_SPEI)
# CreatNC2(lai_SPEI_z)
