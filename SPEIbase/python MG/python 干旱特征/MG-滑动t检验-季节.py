# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 15:02:03 2022

@author: MaYutong
"""

import netCDF4 as nc
from matplotlib import font_manager
import matplotlib.dates as mdate
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
import xarray as xr

#%%
def read_nc():
    global lcc, lat, lon, df
    # global a1, a2, o1, o2
    inpath = (r"E:\ESA\ESA_LUC_2000_SPEI03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        # print(f.variables['lccs_class'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        lcc = (f.variables['lcc'][:])
        
#%%
def read_nc2():
    global spei
    inpath = (r"E:/SPEI_base/data/spei03_MG.nc")
    with nc.Dataset(inpath) as f:
        #print(f.variables.keys())
        
        spei = (f.variables['spei'][:])

#%% mask数组
def mask(x):
    lcc2 = np.array(lcc)
    
    np.place(lcc2, lcc2!=x, 1)
    np.place(lcc2, lcc2==x, 0)
    
    spei_ma = np.empty((480, 30, 50))
    spei_ma = np.ma.array(spei_ma)
    
    for i in range(480):
        a = spei[i, :, :]
        a = ma.masked_array(a, mask=lcc2)
        spei_ma[i, :, :] = a

    spei_ma_ave = spei_ma.mean(axis=(1, 2))
    
    return spei_ma_ave

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
            
#%%
def plot(t_ori, t_move, n, t_thres, a, title):
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
        plt.savefig(rf'E:/SPEI_base/python MG/JPG/{title}.jpg', bbox_inches='tight')
    plt.show()
       
#%%
read_nc()


# lcc2 = np.flip(lcc, axis=0)

read_nc2() 

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
#%%
def season_yr(data):
    tmp_s = np.vstack((data[2:, :], data[:2, :]))
    tmp_sea = []
    for mn1, mn2 in zip(range(0, 12, 3), range(3, 15, 3)):
        tmp_sea.append(tmp_s[mn1:mn2, :])

    tmp_sea = np.array(tmp_sea)
    tmp_sea = tmp_sea.mean(axis=1)

    return tmp_sea
#%%

season = ["Spring", "Summer", "Autumn", "Winter"]

N = [4, 6, 8, 10]
tt = [3.7074, 3.1693, 2.9768, 2.8784]


    
spei_ave = mask(130)

spei_ave_mn = mn_yr(spei_ave)
spei_sea = season_yr(spei_ave_mn)

for j in range(4):
    for k, n in enumerate(N):
        t_move = slidet(spei_sea[j, :], n)
        # print(t_move, "\n")
        a = tip(t_move, tt[k])
        title = f"Moving-T Grassland {season[j]} (n={n})"
        if a==True:
            print(title, "\n")
        plot(spei_sea[j, :], t_move, n, tt[k], a, title)
