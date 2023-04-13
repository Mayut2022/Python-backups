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
    inpath = (f"E:/CRU/TMP_DATA/cru_ts4.06.{a}.{b}.tmp.dat.nc")
    with nc.Dataset(inpath, mode='r') as f:
        '''
        print(f.variables.keys())
        print(f.variables['pre'])
        print(f.variables['time'])
        '''
        #print(f.variables['stn'])
        
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        time = (f.variables['time'][:])
        t = nc.num2date(time, 'days since 1900-01-01 00:00:0.0').data
        tmp = (f.variables['tmp'][:])
        
        return time, tmp

# %%
def region(data):
    tmp_IMglobal = xr.DataArray(data, dims=['t', 'y', 'x'], coords=[
                               time, lat, lon])  # 原tmp-base数据
    tmp_IM = tmp_IMglobal.loc[:, 40:55, 100:135]

    return tmp_IM        

#%%
yr1 = [1981, 1991, 2001, 2011]
yr2 = [1990, 2000, 2010, 2020]
for a, b in zip(yr1, yr2):
    t, _ = read_data_nc1(a, b)
    if a==1981:
        time, tmp = t, _
    else:
        time, tmp = np.hstack((time, t)), np.vstack((tmp, _))
        

#%%
'''
tmp_IM = region1(pre)
lat_IM = tmp_IM.y
lon_IM = tmp_IM.x
'''
#%%生成新的nc文件
def CreatNC(data):
    new_NC = nc.Dataset(r"E:/CRU/TMP_DATA/TMP_CRU_ORIGINAL_81_20.nc", 'w', format='NETCDF4')
    
    new_NC.createDimension('time', 480)
    new_NC.createDimension('lat', 360)
    new_NC.createDimension('lon', 720)
    
    var=new_NC.createVariable('tmp', 'f', ("time","lat","lon"))
    new_NC.createVariable('time', 'f', ("time"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))
    
    new_NC.variables['tmp'][:]=data
    new_NC.variables['time'][:]=time
    new_NC.variables['lat'][:]=lat
    new_NC.variables['lon'][:]=lon
        
    
    #var.lat_range="[40, 55], 30, 精度：0.5, 边界：[40.25, 54.75]"
    #var.lon_range="[100, 135], 70, 精度：0.5, 边界：[100.25, 134.75]"
    var.Fillvalues="9.96921e+36"
    var.time="1981.1-2020.12"
    
    #最后记得关闭文件
    new_NC.close()

# CreatNC(tmp)

#%%生成新的nc文件
def CreatNC2(data):
    new_NC = nc.Dataset(r"E:/CRU/TMP_DATA/TMP_CRU_MONTH_81_20.nc", 'w', format='NETCDF4')
    
    year = np.arange(1, 41, 1)
    month = np.arange(1, 13, 1)
    
    new_NC.createDimension('month', 12)
    new_NC.createDimension('year', 40)
    new_NC.createDimension('lat', 360)
    new_NC.createDimension('lon', 720)
    
    var=new_NC.createVariable('tmp', 'f', ("month","year","lat","lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))
    
    new_NC.variables['tmp'][:]=data
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
    tmp_mn = []
    for mn in range(12):
        tmp_ = []
        for yr in range(40):
            tmp_.append(data[mn])
            mn += 12
        tmp_mn.append(tmp_)
            
    tmp_mn = np.array(tmp_mn)
    
    return tmp_mn

tmp_mn = mn_yr(tmp)

# CreatNC2(tmp_mn)

#%%生成新的nc文件
def CreatNC3(data):
    new_NC = nc.Dataset(r"E:/CRU/TMP_DATA/TMP_CRU_SEASON_81_20.nc", 'w', format='NETCDF4')
    
    year = np.arange(1, 5, 1)
    month = np.arange(1, 13, 1)
    
    new_NC.createDimension('month', 4)
    new_NC.createDimension('year', 40)
    new_NC.createDimension('lat', 360)
    new_NC.createDimension('lon', 720)
    
    var=new_NC.createVariable('tmp', 'f', ("month","year","lat","lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))
    
    new_NC.variables['tmp'][:]=data
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
    tmp_s = np.vstack((data[2:, :], data[:2, :]))
    tmp_sea = []
    for mn1, mn2 in zip(range(0, 12, 3), range(3, 15, 3)):
        tmp_sea.append(tmp_s[mn1:mn2, :])
    
    tmp_sea = np.array(tmp_sea)
    tmp_sea = tmp_sea.mean(axis=1)
    
    return tmp_sea

#%%
tmp_sea = season_yr(tmp_mn)
#CreatNC3(tmp_sea)

#%%生成新的nc文件
def CreatNC4(data, mn):
    new_NC = nc.Dataset(rf"E:/CRU/TMP_DATA/Zscore/TMP_CRU_Zscore_81_20_month{mn}.nc", 'w', format='NETCDF4')
    
    year = np.arange(1, 41, 1)
    
    new_NC.createDimension('year', 40)
    new_NC.createDimension('lat', 360)
    new_NC.createDimension('lon', 720)
    
    var=new_NC.createVariable('tmp', 'f', ("year","lat","lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))
    
    new_NC.variables['tmp'][:]=data
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
    tmp_mn_z = np.zeros((40, 360, 720))
    for mn in range(1, 13): ###########写错了？？？
        for r in range(360):
            if r%30 == 0:
                print(f"columns {r} is done!")
            for c in range(720):
                tmp_mn_z[:, r, c] = preprocessing.scale(data[mn, :, r, c]) #########
         #CreatNC4(tmp_mn_z, mn)
                
#Zscore(tmp_mn)
#%%生成新的nc文件
def CreatNC5(data):
    new_NC = nc.Dataset(r"E:/CRU/TMP_DATA/TMP_CRU_MONTH_ANOM_81_20.nc", 'w', format='NETCDF4')
    
    year = np.arange(1, 41, 1)
    month = np.arange(1, 13, 1)
    
    new_NC.createDimension('month', 12)
    new_NC.createDimension('year', 40)
    new_NC.createDimension('lat', 360)
    new_NC.createDimension('lon', 720)
    
    var=new_NC.createVariable('tmp', 'f', ("month","year","lat","lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))
    
    new_NC.variables['tmp'][:]=data
    new_NC.variables['lat'][:]=lat
    new_NC.variables['lon'][:]=lon
        
    
    #var.lat_range="[40, 55], 30, 精度：0.5, 边界：[40.25, 54.75]"
    #var.lon_range="[100, 135], 70, 精度：0.5, 边界：[100.25, 134.75]"
    var.Fillvalues="9.96921e+36, 缺测值已处理为np.nan"
    var.time="1981.1-2020.12"
    
    #最后记得关闭文件
    new_NC.close()
    
#%%
########## 生成Anom数据
def Anom(data):
    data[data==9.96921e+36]=np.nan
    tmp_mn_ave = np.nanmean(data, axis=1)
    tmp_mn_a = np.zeros((12, 40, 360, 720))
    for mn in range(12):
        print(f"Month{mn+1} is in programming!")
        for yr in range(40):
            for r in range(360):
                if r%30 == 0:
                    print(f"columns {r} is done!")
                for c in range(720):
                    tmp_mn_a[mn, yr, r, c] = data[mn, yr, r, c]-tmp_mn_ave[mn, r, c] #########
    
    return tmp_mn_a

#tmp_mn_anom = Anom(tmp_mn)
#CreatNC5(tmp_mn_anom)
