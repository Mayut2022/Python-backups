# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 11:37:14 2022

@author: MaYutong
"""


import netCDF4 as nc

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
import xarray as xr
import xlsxwriter

from scipy.stats import linregress
from scipy.stats import pearsonr


#%%
def read_nc(inpath):
    global lat, lon, e
    with nc.Dataset(inpath, mode='r') as f:
        
        '''
        print(f.variables.keys())
        print(f.variables['time'])
        print(f.variables['lat'])
        print(f.variables['lon'])
        print(f.variables['E'])
        '''
        t = (f.variables['time'][:])
        
        e = (f.variables['E'][:, :, :])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        
#%%
def read_nc2():
    global spei
    inpath = (r"E:/SPEI_base/data/spei03.nc")
    with nc.Dataset(inpath) as f:
        #print(f.variables.keys())
        
        spei = (f.variables['spei'][960:])
        time = (f.variables['time'][960:])
        t2 = nc.num2date(time, 'days since 1900-01-01 00:00:0.0').data
        lat2 = (f.variables['lat'][:])
        lon2 = (f.variables['lon'][:])

#%%
def spei_month(mn, data):
    e_mn = []
    ind = mn-1
    for i in range(40):
        #print(ind)
        e_mn.append(data[ind, :, :])
        ind+=12
    e_mn = np.array(e_mn)
    return e_mn

#%% n为滑动相关的窗口
def moving_corr(data1, data2, n):
    ind = 0
    s, r0, p = np.zeros((360, 720)), np.zeros((360, 720)), np.zeros((360, 720))
    
    for r in range(360):
        if r%30 == 0:
            print(f"column {r} is done!")
        for c in range(720):
            a = data1[:, r, c] # SPEI 缺测值：1.e+30
            b = data2[:, r, c] # e 缺测值：直接为np.nan
            a[a==1.e+30] = np.nan # 移除缺测值，含缺测值的格点不参与计算
            
            if np.isnan(a).any() or np.isnan(b).any():
                s[r, c], r0[r, c], p[r, c] = np.nan, np.nan, np.nan
            else:
                #print(r, c, "\n", a, "\n", b)
                spei_e = np.hstack((a, b)).reshape(2, 40).T
                spei_e = pd.DataFrame(spei_e.data)
                spei_e.columns = ["SPEI", "GLEAM"]
                corr = spei_e.rolling(n).corr()
                corr_G = corr.iloc[::2, 1].dropna()
                
                if len(corr_G)==0:
                    s[r, c], r0[r, c], p[r, c] = np.nan, np.nan, np.nan
                else:
                    # print(corr, "\n", r, c, "\n")
                    
                    t = np.arange(1, len(corr_G)+1, 1)
                    s[r, c], _, r0[r, c], p[r, c], _  = linregress(t, corr_G)
    
    return s, r0, p

#%%生成新的nc文件
def CreatNC(data1, data2, data3, mn):
    new_NC = nc.Dataset(
        rf'E:/Gleamv3.6a/v3.6a/global/Corr/SPEI_E_81_20_MovingTrend_(n=5)_month{mn}.nc', 
        'w', format='NETCDF4')
    
    new_NC.createDimension('lat', len(lat))
    new_NC.createDimension('lon', len(lon))
    
    new_NC.createVariable('s', 'f', ("lat", "lon"))
    var = new_NC.createVariable('r', 'f', ("lat", "lon"))
    new_NC.createVariable('p', 'f', ("lat", "lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))
    
    new_NC.variables['s'][:]=data1
    new_NC.variables['r'][:]=data2
    new_NC.variables['p'][:]=data3
    new_NC.variables['lat'][:]=lat
    new_NC.variables['lon'][:]=lon
     
    var.description = "1981.1-2020.12 E (actual e) 仅linear插值，其他未处理的标准化距平"
    var.time = f"month(mn) SPEI-E Pearson相关 n=5 滑动相关 回归趋势"
    
    new_NC.close()
    
#%% 逐月相关
read_nc2()

for mn in range(1, 13):
    inpath = rf"E:/Gleamv3.6a/v3.6a/global/Zscore/E_1980-2021_GLEAM_v3.6a_MO_SPEI_0.5X0.5_Zscore_month{mn}.nc"
    read_nc(inpath)
    spei_mn = spei_month(mn, spei)
    s, r, p = moving_corr(spei_mn, e, 5)
    CreatNC(s, r, p, mn)
    
#%% test 部分
'''
data1 = spei_mn
data2 = e

for r in range(141, 142):
    for c in range(231, 232):
        a = data1[:, r, c] # SPEI
        b = data2[:, r, c] # e
        spei_e = np.hstack((a, b)).reshape(2, 40).T
        spei_e = pd.DataFrame(spei_e.data)
        spei_e.columns = ["SPEI", "GLEAM"]
        corr = spei_e.rolling(5).corr()
        corr_G = corr.iloc[::2, 1].dropna()
        t = np.arange(1, len(corr_G)+1, 1)
        s,_,r, p,_  = linregress(t, corr_G)
        print(s, r, p)
'''
#%%
'''
import matplotlib.pyplot as plt

plt.figure(1, dpi=500)
plt.imshow(s, cmap='Blues', vmin=-1, vmax=1, origin="lower")
plt.colorbar(shrink=0.75)
plt.show()
'''