# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 16:33:43 2022

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
from sklearn import preprocessing
import xarray as xr

#%%
def read_nc():
    global lcc, lat, lon
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
    inpath = (rf"E:/SPEI_base/data/spei03_MG.nc")
    with nc.Dataset(inpath) as f:
        #print(f.variables.keys())
        
        spei = (f.variables['spei'][960:])

#%% mask数组
def mask(x, data):
    lcc2 = np.array(lcc)
    
    np.place(lcc2, lcc2!=x, 1)
    np.place(lcc2, lcc2==x, 0)
    
    spei_ma = np.empty((480, 30, 50))
    spei_ma = np.ma.array(spei_ma)
    
    for i in range(480):
        a = data[i, :, :]
        a = ma.masked_array(a, mask=lcc2)
        spei_ma[i, :, :] = a

    spei_ma_ave = np.nanmean(spei_ma, axis=(1, 2))
    
    return spei_ma_ave

        
#%%
def subplot(data1, ax, sy, df):
    t = pd.date_range(f'{sy}', periods=120, freq="MS")
    # ax.axhline(y=std, c="orange", linestyle="--")
    # ax.axhline(y=-std, c="orange", linestyle="--")
    ax.axhline(y=0, c="k", linestyle="--")
    
    for yr in range(1981, 2021):
        d1 = dt.datetime(yr, 3, 1, 0, 0, 0)
        d2 = dt.datetime(yr, 6, 1, 0, 0, 0)
        ax.fill_between([d1, d2], -2, -2.8, facecolor='yellow', alpha=0.4, label="Spring")
        d1 = dt.datetime(yr, 6, 1, 0, 0, 0)
        d2 = dt.datetime(yr, 9, 1, 0, 0, 0)
        ax.fill_between([d1, d2], -2, -2.8, facecolor='orange', alpha=0.4, label="Summer")
        d1 = dt.datetime(yr, 9, 1, 0, 0, 0)
        d2 = dt.datetime(yr, 12, 1, 0, 0, 0)
        ax.fill_between([d1, d2], -2, -2.8, facecolor='blue', alpha=0.4, label="Autumn")
        d1 = dt.datetime(yr, 12, 1, 0, 0, 0)
        d2 = dt.datetime(yr+1, 3, 1, 0, 0, 0)
        ax.fill_between([d1, d2], -2, -2.8, facecolor='k', alpha=0.4, label="Winter")
    
    for y1, m1, y2, m2 in zip(df["SY"], df["SM"], df["EY"], df["EM"]):
        d1 = dt.datetime(y1, m1, 1, 0, 0, 0)
        d2 = dt.datetime(y2, m2, 1, 0, 0, 0)+dt.timedelta(days=31)
        ax.fill_between([d1, d2], 2.5, -2, facecolor='dodgerblue', alpha=0.4, label="Drought") ## 貌似只要是时间格式的x都行
        
    ax.scatter(t, data1, c='b', s=20)
    ax.plot(t, data1, c='b', linewidth=2)
    
    ax.set_ylim(-2.5, 2.5)
    ax.set_yticks(np.arange(-2, 2.1, 1))
    
    mn = pd.to_timedelta("100 days")
    ax.set_xlim(t[0]-mn, t[-1]+mn)
    
    tt = pd.date_range(f'{sy}', periods=10, freq='1AS-JAN')
    ax.set_xticks(tt)
    ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m'))
    ax.tick_params(labelsize=15)
    ax.tick_params(which="major", bottom=1, left=1, length=8)
    
    return ax


def plot(data, df, title):
    global labels
    print(data.max(), data.min())
    fig = plt.figure(1, figsize=(20, 12), dpi=500)
    
    fig.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.93, wspace=None, hspace=0.2)
    axs = fig.subplots(4, 1, sharey=True)
    
    sy = [1981, 1991, 2001, 2011]
    ind = [0, 120, 240, 360]
    for i, ax in enumerate(axs):
        ax = subplot(data[ind[i]:ind[i]+120], ax, sy[i], df)
    
    lines, labels = ax.get_legend_handles_labels()
    fig.legend(lines[:4], labels[:4], loc='upper right', fontsize=15)
    plt.xlabel("years", fontsize=20)
    plt.suptitle(f'{title}', fontsize=25)
    plt.savefig(rf'E:/GIMMS_NDVI/JPG_RG/{title}.jpg', 
                bbox_inches='tight')
    plt.show()

#%%
read_nc()
read_nc2()
spei_ave = mask(130, spei)

df = pd.read_excel("E:/SPEI_base/python MG/MG Grassland干旱统计.xlsx", sheet_name='Grassland')

plot(spei_ave, df, f"MG Grassland Drought Season")
