# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 19:24:00 2022

@author: MaYutong
"""
import matplotlib.pyplot as plt
import netCDF4 as nc
import cftime
import xarray as xr

def read_nc():
    global lat, lon, t, e, time
    inpath = r'E:/Gleamv3.6a/v3.6a/monthly/E_1980-2021_GLEAM_v3.6a_MO.nc'
    with nc.Dataset(inpath, mode='r') as f:
        '''
        print(f.variables.keys())
        print(f.variables['time'])
        print(f.variables['lat'])
        print(f.variables['lon'])
        print(f.variables['E'])
        '''
        time = (f.variables['time'][12:-12])
        t = nc.num2date(time, 'days since 1980-01-31 00:00:00').data
        
        e = (f.variables['E'][12:-12, :, :])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        
#%% Et transpiration
def read_nc3():
    global et, lat3, lon3, t3
    inpath = r'E:/Gleamv3.6a/v3.6a/monthly/Et_1980-2021_GLEAM_v3.6a_MO.nc'
    with nc.Dataset(inpath, mode='r') as f:
        
        print(f.variables.keys())
        print(f.variables['time'])
        print(f.variables['lat'])
        print(f.variables['lon'])
        print(f.variables['Et'])
        
        time = (f.variables['time'][12:-12])
        t3 = nc.num2date(time, 'days since 1980-01-31 00:00:00').data
        
        et = (f.variables['Et'][12:-12, :, :])
        lat3 = (f.variables['lat'][:])
        lon3 = (f.variables['lon'][:])

#%%
def read_nc2():
    global spei, t2, lat2, lon2
    inpath = (r"E:/SPEI_base/data/spei03.nc")
    with nc.Dataset(inpath) as f:
        #print(f.variables.keys())
        
        spei = (f.variables['spei'][960:])
        time = (f.variables['time'][960:])
        t2 = nc.num2date(time, 'days since 1900-01-01 00:00:0.0').data
        lat2 = (f.variables['lat'][:])
        lon2 = (f.variables['lon'][:])

#%% 插值
def GLEAM_SPEI(data):
    e_GLEAM = xr.DataArray(data, dims=['t', 'y', 'x'], coords=[
                               t, lat, lon])  # 原SPEI-base数据
    e_spei = e_GLEAM.interp(t=t, y=lat2, x=lon2, method="linear")

    return e_spei

#%%生成新的nc文件
def CreatNC(data):
    new_NC = nc.Dataset(
        r'E:/Gleamv3.6a/v3.6a/monthly/E_1980-2021_GLEAM_v3.6a_MO_SPEI_0.5X0.5.nc', 
        'w', format='NETCDF4')
    
    new_NC.createDimension('time', 480)
    new_NC.createDimension('lat', len(lat2))
    new_NC.createDimension('lon', len(lon2))
    
    var = new_NC.createVariable('E', 'f', ("time", "lat", "lon"))
    new_NC.createVariable('time', 'f', ("time"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))
    
    new_NC.variables['E'][:]=data
    new_NC.variables['time'][:]=time
    new_NC.variables['lat'][:]=lat2
    new_NC.variables['lon'][:]=lon2
     
    var.description = "1981.1-2020.12 E (actual e) 仅linear插值，其他均未处理，月平均 mm/day"
    var.time = "days since 1980-01-31 00:00:00"
    var.scalar = '将原0.25x0.25精度插值成0.5x0.5, 降尺度 与SPEI数据保持一致'
    
    new_NC.close()
    

def CreatNC2(data):
    new_NC = nc.Dataset(
        r'E:/Gleamv3.6a/v3.6a/monthly/Et_1980-2021_GLEAM_v3.6a_MO_SPEI_0.5X0.5.nc', 
        'w', format='NETCDF4')
    
    new_NC.createDimension('time', 480)
    new_NC.createDimension('lat', len(lat2))
    new_NC.createDimension('lon', len(lon2))
    
    var = new_NC.createVariable('Et', 'f', ("time", "lat", "lon"))
    new_NC.createVariable('time', 'f', ("time"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))
    
    new_NC.variables['Et'][:]=data
    new_NC.variables['time'][:]=time
    new_NC.variables['lat'][:]=lat2
    new_NC.variables['lon'][:]=lon2
     
    var.description = "1981.1-2020.12 Et (蒸腾 Transpiration) 仅linear插值，其他均未处理，月平均 mm/day"
    var.time = "days since 1980-01-31 00:00:00"
    var.scalar = '将原0.25x0.25精度插值成0.5x0.5, 降尺度 与SPEI数据保持一致'
    
    new_NC.close()


#%%
read_nc()

read_nc2()
'''
e_spei = GLEAM_SPEI(e)
CreatNC(e_spei)
'''
#%%
read_nc3()
et_spei = GLEAM_SPEI(et)
CreatNC2(et_spei)
