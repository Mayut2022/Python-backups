# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 11:35:24 2023

@author: MaYutong
"""

import netCDF4 as nc
import numpy as np
import xarray as xr


#%% 读取文件后
def read_lai():
    global lat, lon
    inpath = (r"E:/LAI4g/data_MG/LAI_82_20_MG_SPEI0.5x0.5.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        print(f.variables['lai'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        lai = (f.variables['lai'][:])  ####1982-2020
    return lai



#%%
def mn_yr(data):
    q_mn = []
    for mn in range(12):
        q_ = []
        for yr in range(40):
            q_.append(data[mn])
            mn += 12
        q_mn.append(q_)
            
    q_mn = np.array(q_mn)
    return q_mn


def pre_xarray(band1):
    t = np.arange(40)
    mn = np.arange(1, 13, 1)
    pre=xr.DataArray(band1,dims=['mn', 't', 'y','x'],coords=[mn, t, lat2, lon2])
    pre_MG = pre.loc[:, 1:, 40:55, 100:125]
    lat_MG = pre_MG.y
    lon_MG = pre_MG.x
    pre_MG = np.array(pre_MG)
    return pre_MG, lat_MG, lon_MG


def read_ET():
    global lat2, lon2
    inpath = (r"E:/Gleamv3.6a/Data/global/E_1980-2021_GLEAM_v3.6a_MO_SPEI_0.5X0.5.nc")
    with nc.Dataset(inpath) as f:
        print(f.variables.keys())
        print(f.variables['E'])
        lat2 = (f.variables['lat'][:])
        lon2 = (f.variables['lon'][:])
        et = (f.variables['E'][:])  ####1982-2020
        et_yr = mn_yr(et)
        et_MG, _, _ = pre_xarray(et_yr)
        et_MG = et_MG.transpose(1, 0, 2, 3)
        
    return et_MG


def read_T():
    global lat2, lon2
    inpath = (r"E:/Gleamv3.6a/Data/global/Et_1980-2021_GLEAM_v3.6a_MO_SPEI_0.5X0.5.nc")
    with nc.Dataset(inpath) as f:
        print(f.variables.keys())
        # print(f.variables['lai'])
        lat2 = (f.variables['lat'][:])
        lon2 = (f.variables['lon'][:])
        et = (f.variables['Et'][:])  ####1982-2020
        et_yr = mn_yr(et)
        et_MG, _, _ = pre_xarray(et_yr)
        et_MG = et_MG.transpose(1, 0, 2, 3)
        
    return et_MG


# %%
def CreatNC(data1, data2):
    year = np.arange(1982, 2021, 1)
    month = np.arange(1, 13, 1)

    new_NC = nc.Dataset(
        r"E:/LAI4g/data_MG/WUE_LAI_ET(T)_82_20_MG_SPEI0.5x0.5.nc", 'w', format='NETCDF4')

    new_NC.createDimension('year', 39)
    new_NC.createDimension('month', 12)
    new_NC.createDimension('lat', 30)
    new_NC.createDimension('lon', 50)

    var = new_NC.createVariable('WUE1', 'f', ("year", "month", "lat", "lon"))
    new_NC.createVariable('WUE2', 'f', ("year", "month", "lat", "lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))

    new_NC.variables['WUE1'][:] = data1
    new_NC.variables['WUE2'][:] = data1
    new_NC.variables['lat'][:] = lat
    new_NC.variables['lon'][:] = lon

    var.data = "scale factor和缺测值均已处理，缺测值为np.nan"
    var.units = "LAI:m2/m2 (month mean); ET/T:mm/month "
    var.description = "WUE1:LAI/ET; WUE2:LAI/T; LAI数据来源:GIMMS LAI4g; ET/T数据来源:GLEAMv3.6"

    # 最后记得关闭文件
    new_NC.close()

#%%

lai = read_lai()
et = read_ET()
t = read_T()

#%%
wue1 = lai/et
wue2 = lai/t

#%%
CreatNC(wue1, wue2)
