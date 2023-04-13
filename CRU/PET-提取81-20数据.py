# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 11:40:34 2022

@author: MaYutong
"""

import netCDF4 as nc
import cftime
import xarray as xr
import numpy as np
from sklearn import preprocessing

import warnings
warnings.filterwarnings("ignore")

def read_data_nc1(a, b):
    global lat, lon
    inpath = (f"E:/CRU/pet_DATA/cru_ts4.06.{a}.{b}.pet.dat.nc")
    with nc.Dataset(inpath, mode='r') as f:
        '''
        print(f.variables.keys())
        print(f.variables['pre'])
        print(f.variables['time'])
        '''
        print(f.variables['pet'])
        
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        time = (f.variables['time'][:])
        t = nc.num2date(time, 'days since 1900-01-01 00:00:0.0').data
        pet = (f.variables['pet'][:])
        
        return time, pet

# %%
def region(data):
    pet_IMglobal = xr.DataArray(data, dims=['t', 'y', 'x'], coords=[
                               time, lat, lon])  # 原pet-base数据
    pet_IM = pet_IMglobal.loc[:, 40:55, 100:135]

    return pet_IM        

#%%
yr1 = [1981, 1991, 2001, 2011]
yr2 = [1990, 2000, 2010, 2020]
for a, b in zip(yr1, yr2):
    t, _ = read_data_nc1(a, b)
    if a==1981:
        time, pet = t, _
    else:
        time, pet = np.hstack((time, t)), np.vstack((pet, _))
        

#%%
'''
pet_IM = region1(pre)
lat_IM = pet_IM.y
lon_IM = pet_IM.x
'''
#%%生成新的nc文件
def CreatNC(data):
    new_NC = nc.Dataset(r"E:/CRU/PET_DATA/pet_CRU_ORIGINAL_81_20.nc", 'w', format='NETCDF4')
    
    new_NC.createDimension('time', 480)
    new_NC.createDimension('lat', 360)
    new_NC.createDimension('lon', 720)
    
    var=new_NC.createVariable('pet', 'f', ("time","lat","lon"))
    new_NC.createVariable('time', 'f', ("time"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))
    
    new_NC.variables['pet'][:]=data
    new_NC.variables['time'][:]=time
    new_NC.variables['lat'][:]=lat
    new_NC.variables['lon'][:]=lon
        
    
    #var.lat_range="[40, 55], 30, 精度：0.5, 边界：[40.25, 54.75]"
    #var.lon_range="[100, 135], 70, 精度：0.5, 边界：[100.25, 134.75]"
    var.units="mm/day potential evapotranspiration"
    var.Fillvalues="9.96921e+36"
    var.time="1981.1-2020.12"
    
    #最后记得关闭文件
    new_NC.close()

#CreatNC(pet)

#%%生成新的nc文件
def CreatNC2(data):
    new_NC = nc.Dataset(r"E:/CRU/pet_DATA/pet_CRU_MONTH_81_20.nc", 'w', format='NETCDF4')
    
    year = np.arange(1, 41, 1)
    month = np.arange(1, 13, 1)
    
    new_NC.createDimension('month', 12)
    new_NC.createDimension('year', 40)
    new_NC.createDimension('lat', 360)
    new_NC.createDimension('lon', 720)
    
    var=new_NC.createVariable('pet', 'f', ("month","year","lat","lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))
    
    new_NC.variables['pet'][:]=data
    new_NC.variables['lat'][:]=lat
    new_NC.variables['lon'][:]=lon
        
    
    #var.lat_range="[40, 55], 30, 精度：0.5, 边界：[40.25, 54.75]"
    #var.lon_range="[100, 135], 70, 精度：0.5, 边界：[100.25, 134.75]"
    var.Fillvalues="9.96921e+36"
    var.time="1981.1-2020.12"
    
    #最后记得关闭文件
    new_NC.close()
    
#%% 480(输入data) -> 月 年
def mn_yr(data):
    pet_mn = []
    for mn in range(12):
        pet_ = []
        for yr in range(40):
            pet_.append(data[mn])
            mn += 12
        pet_mn.append(pet_)
            
    pet_mn = np.array(pet_mn)
    
    return pet_mn

pet_mn = mn_yr(pet)

# CreatNC2(pet_mn)

#%%生成新的nc文件
def CreatNC3(data):
    new_NC = nc.Dataset(r"E:/CRU/pet_DATA/pet_CRU_SEASON_81_20.nc", 'w', format='NETCDF4')
    
    year = np.arange(1, 5, 1)
    month = np.arange(1, 13, 1)
    
    new_NC.createDimension('month', 4)
    new_NC.createDimension('year', 40)
    new_NC.createDimension('lat', 360)
    new_NC.createDimension('lon', 720)
    
    var=new_NC.createVariable('pet', 'f', ("month","year","lat","lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))
    
    new_NC.variables['pet'][:]=data
    new_NC.variables['lat'][:]=lat
    new_NC.variables['lon'][:]=lon
        
    
    #var.lat_range="[40, 55], 30, 精度：0.5, 边界：[40.25, 54.75]"
    #var.lon_range="[100, 135], 70, 精度：0.5, 边界：[100.25, 134.75]"
    var.Fillvalues="9.96921e+36"
    var.time="1981.1-2020.12"
    var.sort= "春 夏 秋 冬"
    
    #最后记得关闭文件
    new_NC.close()
    
#%% 480 -> 月 年（输入data）-> 季节 年
def season_yr(data):
    pet_s = np.vstack((data[2:, :], data[:2, :]))
    pet_sea = []
    for mn1, mn2 in zip(range(0, 12, 3), range(3, 15, 3)):
        pet_sea.append(pet_s[mn1:mn2, :])
    
    pet_sea = np.array(pet_sea)
    pet_sea = pet_sea.mean(axis=1)
    
    return pet_sea

#%%
pet_sea = season_yr(pet_mn)
#CreatNC3(pet_sea)

#%%生成新的nc文件
def CreatNC4(data, mn):
    new_NC = nc.Dataset(rf"E:/CRU/pet_DATA/Zscore/pet_CRU_Zscore_81_20_month{mn}.nc", 'w', format='NETCDF4')
    
    year = np.arange(1, 41, 1)
    
    new_NC.createDimension('year', 40)
    new_NC.createDimension('lat', 360)
    new_NC.createDimension('lon', 720)
    
    var=new_NC.createVariable('pet', 'f', ("year","lat","lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))
    
    new_NC.variables['pet'][:]=data
    new_NC.variables['lat'][:]=lat
    new_NC.variables['lon'][:]=lon
        
    
    #var.lat_range="[40, 55], 30, 精度：0.5, 边界：[40.25, 54.75]"
    #var.lon_range="[100, 135], 70, 精度：0.5, 边界：[100.25, 134.75]"
    var.Fillvalues="9.96921e+36"
    var.time="1981.1-2020.12 'days since 1900-01-01 00:00:0.0'"
    
    #最后记得关闭文件
    new_NC.close()
    
#%% 480 -> 归一化值 是不是写错了？？？
def Zscore(data):
    pet_mn_z = np.zeros((40, 360, 720))
    for mn in range(1, 13): ###########写错了？？？
        for r in range(360):
            if r%30 == 0:
                print(f"columns {r} is done!")
            for c in range(720):
                pet_mn_z[:, r, c] = preprocessing.scale(data[mn, :, r, c]) #########
         #CreatNC4(pet_mn_z, mn)
                
#Zscore(pet_mn)
#%%生成新的nc文件
def CreatNC5(data):
    new_NC = nc.Dataset(r"E:/CRU/PET_DATA/pet_CRU_MONTH_ANOM_81_20.nc", 'w', format='NETCDF4')
    
    year = np.arange(1, 41, 1)
    month = np.arange(1, 13, 1)
    
    new_NC.createDimension('month', 12)
    new_NC.createDimension('year', 40)
    new_NC.createDimension('lat', 360)
    new_NC.createDimension('lon', 720)
    
    var=new_NC.createVariable('pet', 'f', ("month","year","lat","lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))
    
    new_NC.variables['pet'][:]=data
    new_NC.variables['lat'][:]=lat
    new_NC.variables['lon'][:]=lon
        
    
    #var.lat_range="[40, 55], 30, 精度：0.5, 边界：[40.25, 54.75]"
    #var.lon_range="[100, 135], 70, 精度：0.5, 边界：[100.25, 134.75]"
    var.units="mm/day potential evapotranspiration"
    var.Fillvalues="9.96921e+36, 缺测值已处理为np.nan"
    var.time="1981.1-2020.12"
    
    #最后记得关闭文件
    new_NC.close()
    
#%%
########## 生成Anom数据
def Anom(data):
    data[data==9.96921e+36]=np.nan
    pet_mn_ave = np.nanmean(data, axis=1)
    pet_mn_a = np.zeros((12, 40, 360, 720))
    for mn in range(12):
        print(f"Month{mn+1} is in programming!")
        for yr in range(40):
            for r in range(360):
                if r%30 == 0:
                    print(f"columns {r} is done!")
                for c in range(720):
                    pet_mn_a[mn, yr, r, c] = data[mn, yr, r, c]-pet_mn_ave[mn, r, c] #########
    
    return pet_mn_a

pet_mn_anom = Anom(pet_mn)
CreatNC5(pet_mn_anom)
