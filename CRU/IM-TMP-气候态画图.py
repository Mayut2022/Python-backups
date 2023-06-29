# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 17:53:55 2022

@author: MaYutong
"""

import netCDF4 as nc
import math
from matplotlib import font_manager
import matplotlib.dates as mdate
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
import xarray as xr


import warnings
warnings.filterwarnings("ignore")

#%%
def read_data_nc1():
    global lat_g, lon_g
    inpath = (f"E:/CRU/TMP_DATA/TMP_CRU_MONTH_81_20.nc")
    with nc.Dataset(inpath, mode='r') as f:
        '''
        print(f.variables.keys())
        print(f.variables['pre'])
        print(f.variables['time'])
        '''
        #print(f.variables['stn'])
        
        lat_g = (f.variables['lat'][:])
        lon_g = (f.variables['lon'][:])
        tmp = (f.variables['tmp'][:])
        
        return tmp
    
def read_data_nc2():
    global lat_g, lon_g
    inpath = (f"E:/CRU/TMP_DATA/TMP_CRU_SEASON_81_20.nc")
    with nc.Dataset(inpath, mode='r') as f:
        '''
        print(f.variables.keys())
        print(f.variables['pre'])
        print(f.variables['time'])
        '''
        #print(f.variables['stn'])
        
        tmp = (f.variables['tmp'][:])
        
        return tmp
    
def region(data):
    year = np.arange(1, 41, 1)
    month = np.arange(1, 13, 1)
    
    tmp_IMglobal = xr.DataArray(data, dims=['mn', "yr", 'y', 'x'], coords=[
                               month, year, lat_g, lon_g])  # 原tmp-base数据
    tmp_IM = tmp_IMglobal.loc[:, :, 37:46, 105:125]

    return tmp_IM      

def region2(data):
    year = np.arange(1, 41, 1)
    month = np.arange(1, 5, 1)
    
    tmp_IMglobal = xr.DataArray(data, dims=['mn', "yr", 'y', 'x'], coords=[
                               month, year, lat_g, lon_g])  # 原tmp-base数据
    tmp_IM = tmp_IMglobal.loc[:, :, 37:46, 105:125]

    return tmp_IM      

def read_nc():
    global lcc, lat, lon
    # global a1, a2, o1, o2
    inpath = (r"E:\ESA\ESA_LUC_2000_SPEI03_IM.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        # print(f.variables['lccs_class'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        lcc = (f.variables['lcc'][:])

#%% mask数组
def mask(x, data):
    lcc2 = np.array(lcc)
    
    np.place(lcc2, lcc2!=x, 1)
    np.place(lcc2, lcc2==x, 0)
    
    spei_ma = np.empty((4, 40, 18, 40))
    spei_ma = np.ma.array(spei_ma)
    
    for i in range(4):
        for j in range(40):
            a = data[i, j, :, :]
            a = ma.masked_array(a, mask=lcc2)
            spei_ma[i, j, :, :] = a

    spei_ma_ave = spei_ma.mean(axis=(2, 3))
    
    return spei_ma_ave

def mask2(x, data):
    lcc2 = np.array(lcc)
    
    np.place(lcc2, lcc2!=x, 1)
    np.place(lcc2, lcc2==x, 0)
    
    spei_ma = np.empty((12, 40, 18, 40))
    spei_ma = np.ma.array(spei_ma)
    
    for i in range(12):
        for j in range(40):
            a = data[i, j, :, :]
            a = ma.masked_array(a, mask=lcc2)
            spei_ma[i, j, :, :] = a

    spei_ma_ave = spei_ma.mean(axis=(2, 3))
    
    return spei_ma_ave

#%%
def plot_T(tmp, title):
    fig = plt.figure(figsize=(12, 6), dpi=500)
    fig.subplots_adjust(left=0.05, bottom=0.15, right=0.95,
                        top=0.92, wspace=None, hspace=0.1)
    ax = fig.add_subplot(111)

    t1 = pd.date_range(f'1981', periods=40, freq="YS")
    ax.plot(t1, tmp, 'k', label='Original')
    ax.scatter(t1, tmp, color='k')
    ax.axhline(y=tmp.mean(), c="orange", linestyle="--")

    ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y'))
    
    l = math.floor(tmp.min())
    r = math.ceil(tmp.max())
    ax.set_ylim(l-1, r+1)     
    ax.set_yticks(np.arange(l, r+1, 1))
    
    ax.set_ylabel("Units: Degrees Celsius", fontsize=15)
    ax.set_xlabel("Years (Mean 2 m temperature)", fontsize=15)
    ax.tick_params(labelsize=15)
    #ax.text(2020, l-1.1, "Mean 2 m temperature", c="gray")

    plt.legend(loc="upper right", fontsize=15)
    plt.suptitle(f'{title}', fontsize=20)
    plt.savefig(rf'E:\CRU\TMP_JPG\{title}.jpg', bbox_inches='tight')
    plt.show()
    

#%%
def slidet(inputdata, step):
    inputdata = np.array(inputdata)
    n = inputdata.shape[0]
    n1 = step    #n1, n2为子序列长度，需调整
    n2 = step
    t = np.zeros(n)
    for i in range (step, n-step-1):
        x1 = inputdata[i-step : i]
        x2 = inputdata[i : i+step]
        x1_mean = np.nanmean(inputdata[i-step : i])   
        x2_mean = np.nanmean(inputdata[i : i+step])
        s1 = np.nanvar(inputdata[i-step : i])          
        s2 = np.nanvar(inputdata[i : i+step])
        s = np.sqrt((n1 * s1 + n2 * s2) / (n1 + n2 - 2))
        t[i] = (x2_mean - x1_mean) / (s * np.sqrt(1/n1+1/n2))
    t[:step]=np.nan  
    t[n-step+1:]=np.nan 
    
    return t    

def tip(data, thre):
    a=False
    year = np.arange(1981, 2021, 1)
    for d, yr in zip(data, year):
        if np.isnan(d)==False:
            if d>thre or d<-thre:
                print(yr)
                a=True
    return a

def plot_Tmove(t_ori, t_move, n, t_thres, a, title):
    fig = plt.figure(figsize=(12, 6), dpi=500)
    fig.subplots_adjust(left=0.05, bottom=0.15, right=0.95, top=0.92, wspace=None, hspace=0.1)
    ax = fig.add_subplot(111)
    
    t1 = pd.date_range(f'1981', periods=40, freq="YS")
    t2 = pd.date_range(f'{1981+n}', periods=40-2*n, freq="YS")
    ax.plot(t1, t_ori, 'k', label='Original')
    ax.plot(t1, t_move, 'orange', label='Moving t test')
    ax.scatter(t1, t_ori, color='k')
    ax.scatter(t1, t_move, color='orange')
    
    ax.axhline(y=t_thres, c="b", linestyle="--")
    ax.axhline(y=-t_thres, c="b", linestyle="--")
    
    ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y'))
    
    ax.set_ylim(-6, 6)
    ax.set_yticks(np.arange(-5, 5.1, 1))
    #ax.set_ylabel("percent", fontsize=15)
    ax.tick_params(labelsize=15)
    
    plt.legend(loc="upper right", fontsize=15)
    plt.suptitle(f'{title}', fontsize=20)
    if a==True:
        plt.savefig(rf'E:/CRU/TMP_JPG/滑动t检验/{title}.jpg', bbox_inches='tight')
    plt.show()

#%%
df = pd.read_excel('E:/ERA5/每月天数.xlsx')
mn_str = df['月份']
#%%
tmp = read_data_nc1()
tmp_IM = region(tmp)
lat_IM = tmp_IM.y
lon_IM = tmp_IM.x
read_nc()
tmp_sea = read_data_nc2()
tmp_sea_IM = region2(tmp_sea)

#%%
tmp_IM_ave = mask2(130, tmp_IM)
for mn in range(12):
    print(tmp_IM_ave[mn, :].max(), tmp_IM_ave[mn, :].min())
    #plot_T(tmp_IM_ave[mn, :], f"TMP 81-20 {mn_str[mn]}")
    
    
#%%
season = ["Spring", "Summer", "Autumn", "Winter"]
tmp_sea_IM_ave = mask(130, tmp_sea_IM)
for mn in range(4):
    print(tmp_sea_IM_ave[mn, :].max(), tmp_sea_IM_ave[mn, :].min())
    plot_T(tmp_sea_IM_ave[mn, :], f"TMP 81-20 {season[mn]}")
    
#%% 突变
N = [4, 6, 8, 10]
tt = [3.7074, 3.1693, 2.9768, 2.8784]
for mn in range(12):
    for k, n in enumerate(N):
        t_move = slidet(tmp_IM_ave[mn, :], n)
        a = tip(t_move, tt[k])
        title = f"TMP Grassland {mn_str[mn]} (n={n})"
        if a==True:
            print(title, "\n")
        plot_Tmove(tmp_IM_ave[mn, :], t_move, n, tt[k], a, title)
    
    
    
    
    
#%%
'''
import matplotlib.pyplot as plt

plt.figure(3, dpi=500)
plt.imshow(tmp_IM[5, 0, :, :], cmap='Reds', origin="lower")
plt.colorbar(shrink=0.75)
plt.show()

import matplotlib.pyplot as plt

plt.figure(3, dpi=500)
plt.imshow(lcc, cmap='Set3', origin="lower")
plt.colorbar(shrink=0.75)
plt.show()
'''