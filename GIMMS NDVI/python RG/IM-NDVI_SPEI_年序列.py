# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 09:24:37 2022

@author: MaYutong
"""

import datetime as dt
import netCDF4 as nc
from matplotlib import cm
from matplotlib import font_manager
import matplotlib.dates as mdate
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
import pwlf
import xarray as xr
from scipy import optimize
#%%
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

#%%
def read_nc2(inpath):
    global spei, t
    
    with nc.Dataset(inpath) as f:
        #print(f.variables.keys())
        
        spei = (f.variables['spei'][80:])

        
#%%
def read_nc3():
    global ndvi, lat2, lon2
    inpath = r"E:/GIMMS_NDVI/data_RG/NDVI_82_15_RG_SPEI0.5x0.5.nc"
    with nc.Dataset(inpath) as f:
        #print(f.variables.keys())
        
        ndvi = (f.variables['ndvi'][:])
        lat2 = (f.variables['lat'][:])
        lon2 = (f.variables['lon'][:])

#%%
def ndvi_xarray(band1):
    t = np.arange(34)
    ndvi=xr.DataArray(band1,dims=['t','y','x'],coords=[t, lat2, lon2])
    ndvi_IM = ndvi.loc[:, 37:46, 105:125]
    lat_IM = ndvi_IM.y
    lon_IM = ndvi_IM.x
    ndvi_IM = np.array(ndvi_IM)
    return ndvi_IM, lat_IM, lon_IM

#%% mask数组
def mask(x, data):
    lcc2 = np.array(lcc)
    
    np.place(lcc2, lcc2!=x, 1)
    np.place(lcc2, lcc2==x, 0)
    
    spei_ma = np.empty((40, 18, 40))
    spei_ma = np.ma.array(spei_ma)
    
    for i in range(40):
        a = data[i, :, :]
        a = ma.masked_array(a, mask=lcc2)
        spei_ma[i, :, :] = a

    spei_ma_ave = spei_ma.mean(axis=(1, 2))
    
    return spei_ma_ave

#%% mask数组
def mask2(x, data):
    lcc2 = np.array(lcc)
    
    np.place(lcc2, lcc2!=x, 1)
    np.place(lcc2, lcc2==x, 0)
    
    spei_ma = np.empty((34, 18, 40))
    spei_ma = np.ma.array(spei_ma)
    
    for i in range(34):
        a = data[i, :, :]
        a = ma.masked_array(a, mask=lcc2)
        spei_ma[i, :, :] = a

    spei_ma_ave = np.nanmean(spei_ma, axis=(1, 2))
    
    return spei_ma_ave.data

#%%
def plot(data, title):
    fig = plt.figure(figsize=(12, 6), dpi=500)
    
    fig.subplots_adjust(left=0.05, bottom=0.15, right=0.95, top=0.9, wspace=None, hspace=0.1)
    ax = fig.add_subplot(111)
    
    t = pd.date_range('1981', periods=40, freq="YS")
    
    
    ax.axhline(y=0, c="k", linestyle="-")
    '''
    ax.axhline(y=-std, c="orange", linestyle="--")
    
    
    for y1, m1, y2, m2 in zip(df["SY"], df["SM"], df["EY"], df["EM"]):
        d1 = dt.datetime(y1, m1, 1, 0, 0, 0)
        d2 = dt.datetime(y2, m2, 1, 0, 0, 0)+dt.timedelta(days=35)
        ax.fill_between([d1, d2], 2.8, -2.8, facecolor='dodgerblue', alpha=0.4) ## 貌似只要是时间格式的x都行
    '''
    #渐变色柱状图
    #归一化
    norm = plt.Normalize(-1.3, 1.3) #值的范围
    norm_values = norm(data)
    map_vir = cm.get_cmap(name='bwr_r')
    colors = map_vir(norm_values)
    ax.bar(t, data, color=colors, width=250)
    #ax.plot(t, data, color="k")
    
    '''
    #曲线拟合
    x = np.arange(1, 41, 1)
    z1 = np.polyfit(x, data, 1)   
    p1 = np.poly1d(z1) 
    print(p1)
    data_pred = p1(x)
    ax.plot(t, data_pred, c="k", linestyle="--")
    '''
    
    #### 参数设置
    ax.set_ylim(-1.3, 1.3)
    ax.set_yticks(np.arange(-1, 1.1, 0.5))
    
    
    mn = pd.to_timedelta("200 days")
    ax.set_xlim(t[0]-mn, t[-1]+mn)
    
    
    tt = pd.date_range('1982', periods=10, freq='4AS-JAN')
    ax.set_xticks(tt)
    ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y'))
    ax.tick_params(labelsize=20)
    ax.tick_params(which="major", bottom=1, left=1, length=8)
    
    plt.xlabel("years", fontsize=20)
    plt.ylabel("SPEI03 Annual", fontsize=20)
    plt.suptitle(f'{title}', fontsize=25)
    #plt.savefig(rf'E:/SPEI_base/JPG_IM/{title}.jpg', bbox_inches = 'tight')
    plt.show()

#%%
def plot2(data, title):
    fig = plt.figure(figsize=(12, 6), dpi=500)
    
    fig.subplots_adjust(left=0.05, bottom=0.15, right=0.95, top=0.9, wspace=None, hspace=0.1)
    ax = fig.add_subplot(111)
    
    t = pd.date_range('1982', periods=34, freq="YS")
    
    ax.axhline(y=0, c="k", linestyle="-")
    
    '''
    ax.axhline(y=-std, c="orange", linestyle="--")
    
    
    for y1, m1, y2, m2 in zip(df["SY"], df["SM"], df["EY"], df["EM"]):
        d1 = dt.datetime(y1, m1, 1, 0, 0, 0)
        d2 = dt.datetime(y2, m2, 1, 0, 0, 0)+dt.timedelta(days=35)
        ax.fill_between([d1, d2], 2.8, -2.8, facecolor='dodgerblue', alpha=0.4) ## 貌似只要是时间格式的x都行
    '''
    
    #渐变色柱状图
    #归一化
    norm = plt.Normalize(0, 0.5) #值的范围
    norm_values = norm(data)
    map_vir = cm.get_cmap(name='Greens')
    colors = map_vir(norm_values)
    ax.bar(t, data, color=colors, width=250)
    #ax.plot(t, data, color="k")
    
    '''
    #曲线拟合
    x = np.arange(1, 35, 1)
    z1 = np.polyfit(x, data, 1)   
    p1 = np.poly1d(z1) 
    print(p1)
    data_pred = p1(x)
    ax.plot(t, data_pred, c="k", linestyle="--")
    '''
    
    #### 参数设置
    ax.set_ylim(0, 0.6)
    ax.set_yticks(np.arange(0, 0.51, 0.25))
    
    
    mn = pd.to_timedelta("200 days")
    ax.set_xlim(t[0]-mn, t[-1]+mn)
    
    
    tt = pd.date_range('1982', periods=9, freq='4AS-JAN')
    ax.set_xticks(tt)
    ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y'))
    ax.tick_params(labelsize=20)
    ax.tick_params(which="major", bottom=1, left=1, length=8)
    
    plt.xlabel("years", fontsize=20)
    plt.ylabel("NDVI", fontsize=20)
    plt.suptitle(f'{title}', fontsize=25)
    plt.savefig(rf'E:/GIMMS_NDVI/JPG_RG/{title}.jpg', bbox_inches = 'tight')
    plt.show()
    
#%%
read_nc()
# lcc2 = np.flip(lcc, axis=0) lcc和SPEI都是反的，不需要翻转
#%%
def read_plot(spei):
    global spei_ave
    spei_ave = mask(130, spei)
    print(spei_ave.max(), spei_ave.min(), spei_ave.std())
    plot(spei_ave,  f"Grassland SPEI03 Annual 81-20 Bar")
    
inpath = (rf"E:/SPEI_base/data/spei03_IM_annual.nc")
read_nc2(inpath)
read_plot(spei)
#%%
def read_plot2(ndvi):
    
    ndvi = np.nanmean(ndvi, axis=1)
    ndvi_IM, lat_IM, lon_IM = ndvi_xarray(ndvi)
    ndvi_ave = mask2(130, ndvi_IM)
    print(ndvi_ave.max(), ndvi_ave.min(), ndvi_ave.std())
    plot2(ndvi_ave,  f"Grassland NDVI Annual 81-20 Bar")

read_nc3()
read_plot2(ndvi)

