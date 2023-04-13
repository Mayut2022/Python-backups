# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 18:05:11 2022

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
def read_nc2(inpath):
    global gpp, lat2, lon2
    with nc.Dataset(inpath) as f:
        #print(f.variables.keys())
        
        gpp = (f.variables['GPP'][:])
        lat2 = (f.variables['lat'][:])
        lon2 = (f.variables['lon'][:])
        
    return gpp
        
#%%
def read_nc3(inpath):
    global gpp_a
    with nc.Dataset(inpath) as f:
        #print(f.variables.keys())
        
        gpp_a = (f.variables['GPP'][:])
        lat2 = (f.variables['lat'][:])
        lon2 = (f.variables['lon'][:])
        
    return gpp_a

#%%
def read_nc4(inpath):
    global wue
    with nc.Dataset(inpath) as f:
        #print(f.variables.keys())
        
        wue = (f.variables['WUE'][:])
        lat2 = (f.variables['lat'][:])
        lon2 = (f.variables['lon'][:])
        
    return wue

#%%
def read_nc5(inpath):
    global wue_a
    with nc.Dataset(inpath) as f:
        #print(f.variables.keys())
        
        wue_a = (f.variables['WUE'][:])
        lat2 = (f.variables['lat'][:])
        lon2 = (f.variables['lon'][:])
        
    return wue_a

#%%
def sif_xarray(band1):
    mn = np.arange(12)
    sif=xr.DataArray(band1, dims=['mn', 'y','x'],coords=[mn, lat2, lon2])
    
    sif_MG = sif.loc[:, 40:55, 100:125]
    return np.array(sif_MG)

#%% mask数组
def mask(x, data):
    lcc2 = np.array(lcc)
    
    np.place(lcc2, lcc2!=x, 1)
    np.place(lcc2, lcc2==x, 0)
    
    spei_ma = np.empty((12, 30, 50))
    spei_ma = np.ma.array(spei_ma)
    
    
    for l in range(12):
        a = data[l, :, :]
        a = ma.masked_array(a, mask=lcc2)
        spei_ma[l, :, :] = a

    spei_ma_ave = np.nanmean(spei_ma, axis=(1, 2))
    
    return spei_ma_ave

#%%
read_nc()

def exact_data1():
    for yr in range(1982, 2019):
        inpath =  rf"E:/GLASS-GPP/Month RG/GLASS_GPP_RG_SPEI_0.5X0.5_{yr}.nc"
        data = read_nc2(inpath)
        
        data_MG = sif_xarray(data)
        data_ave = mask(130, data_MG)
        if yr == 1982:
            data_all = data_ave
        else:
            data_all = np.hstack((data_all, data_ave))
        
    return data_all

def exact_data2():
    for yr in range(1982, 2019):
        inpath2 =  rf"E:/GLASS-GPP/Month RG/GLASS_GPP_RG_Anomaly_SPEI_0.5X0.5_{yr}.nc"
        data = read_nc3(inpath2)
        
        data_MG = sif_xarray(data)
        data_ave = mask(130, data_MG)
        if yr == 1982:
            data_all = data_ave
        else:
            data_all = np.hstack((data_all, data_ave))
        
    return data_all

def exact_data3():
    for yr in range(1982, 2019):
        inpath3 =  rf"E:/GLASS-GPP/Month RG WUE/GLASS_WUE_RG_SPEI_0.5X0.5_{yr}.nc"
        data = read_nc4(inpath3)
        
        data_MG = sif_xarray(data)
        data_ave = mask(130, data_MG)
        if yr == 1982:
            data_all = data_ave
        else:
            data_all = np.hstack((data_all, data_ave))
        
    return data_all

def exact_data4():
    for yr in range(1982, 2019):
        inpath4 =  rf"E:/GLASS-GPP/Month RG WUE/GLASS_WUE_RG_Anomaly_SPEI_0.5X0.5_{yr}.nc"
        data = read_nc5(inpath4)
        
        data_MG = sif_xarray(data)
        data_ave = mask(130, data_MG)
        if yr == 1982:
            data_all = data_ave
        else:
            data_all = np.hstack((data_all, data_ave))
        
    return data_all

#%% GPP
gpp_MG = exact_data1()
gpp_a_MG = exact_data2()
wue_MG = exact_data3()
wue_a_MG = exact_data4()
    
#%% 
'''
import matplotlib.pyplot as plt
a = gpp[7, :, :]
plt.figure(1, dpi=500)
plt.imshow(a, cmap="Blues")
plt.colorbar(shrink=0.75)
plt.show()
'''

#%% 画图
def subplot(data1, data2, ax, sy, df):
    t = pd.date_range(f'{sy}', periods=240, freq="MS")
    # ax.axhline(y=std, c="orange", linestyle="--")
    # ax.axhline(y=-std, c="orange", linestyle="--")
    #ax.axhline(y=0, c="k", linestyle="--")
    
    for y1, m1, y2, m2 in zip(df["SY"], df["SM"], df["EY"], df["EM"]):
        d1 = dt.datetime(y1, m1, 1, 0, 0, 0)
        d2 = dt.datetime(y2, m2, 1, 0, 0, 0)+dt.timedelta(days=35)
        ax.fill_between([d1, d2], 200, -200, facecolor='dodgerblue', alpha=0.4) ## 貌似只要是时间格式的x都行
        
    ax.scatter(t, data1, c='k', s=20)
    ax.plot(t, data1, c='k', label="GPP GLASS", linewidth=2)
    #ax.scatter(t, data2, c='Green')
    
    ax2 = ax.twinx()
    ax2.bar(t, data2, color='orange', label="GPP Anomaly", width=25, alpha=0.7)
    ax2.axhline(y=0, c="orange", linestyle="--")
    
    ########
    ax.set_ylim(-5, 155)
    ax.set_yticks(np.arange(0, 151, 30))
    
    ax2.set_ylim(-30, 30)
    ax2.set_yticks(np.arange(-30, 31, 10))
    
    mn = pd.to_timedelta("100 days")
    ax.set_xlim(t[0]-mn, t[-1]+mn)
    
    tt = pd.date_range(f'{sy}', periods=10, freq='2AS-JAN')
    ax.set_xticks(tt)
    ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m'))
    ax.tick_params(labelsize=20)
    ax.tick_params(which="major", bottom=1, left=1, length=8)
    ax2.tick_params(labelsize=20)
    ax2.tick_params(which="major", bottom=1, left=0, length=8)
    
    #### 添加图例，本质上line1为list
    line1, label1 = ax.get_legend_handles_labels()
    line2, label2 = ax2.get_legend_handles_labels()
    lines = line1+line2
    labels = label1+label2
    
    
    return ax, lines, labels


def plot(data1, data2, df, title):
    print(np.nanmax(data1), np.nanmin(data1))
    print(np.nanmax(data2), np.nanmin(data2))
    
    fig = plt.figure(1, figsize=(22, 12), dpi=500)
    
    fig.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.93, wspace=None, hspace=0.12)
    axs = fig.subplots(2, 1, sharey=True)
    
    a, b = np.zeros(480), np.zeros(480)
    a[0:12], a[12:456], a[456:] = np.nan, data1, np.nan
    b[0:12], b[12:456], b[456:] = np.nan, data2, np.nan
    
    data1 = a
    data2 = b
    
    axs[0], lines, labels = subplot(data1[:240], data2[:240], axs[0], 1981, df)
    axs[1], lines, labels = subplot(data1[240:], data2[240:], axs[1], 2001, df)
    
    fig.legend(lines, labels, loc = 'upper right', bbox_to_anchor=(0.95, 1), fontsize=20)
    
    plt.xlabel("years", fontsize=25)
    plt.suptitle(f'{title}', fontsize=30)
    plt.savefig(rf'E:/GLASS-GPP/JPG MG/{title}.jpg', 
                bbox_inches='tight')
    plt.show()

#%% 画图
def subplot2(data1, data2, ax, sy, df):
    t = pd.date_range(f'{sy}', periods=240, freq="MS")
    # ax.axhline(y=std, c="orange", linestyle="--")
    # ax.axhline(y=-std, c="orange", linestyle="--")
    #ax.axhline(y=0, c="k", linestyle="--")
    
    for y1, m1, y2, m2 in zip(df["SY"], df["SM"], df["EY"], df["EM"]):
        d1 = dt.datetime(y1, m1, 1, 0, 0, 0)
        d2 = dt.datetime(y2, m2, 1, 0, 0, 0)+dt.timedelta(days=35)
        ax.fill_between([d1, d2], 200, -200, facecolor='dodgerblue', alpha=0.4) ## 貌似只要是时间格式的x都行
        
    ax.scatter(t, data1, c='k', s=20)
    ax.plot(t, data1, c='k', label="WUE", linewidth=2)
    #ax.scatter(t, data2, c='Green')
    
    ax2 = ax.twinx()
    ax2.bar(t, data2, color='#7B68EE', label="WUE Anomaly", width=25, alpha=0.7)
    ax2.axhline(y=0, c="#7B68EE", linestyle="--")
    
    ########
    ax.set_ylim(-1, 3)
    ax.set_yticks(np.arange(-1, 3.1, 1))
    
    ax2.set_ylim(-1, 1)
    ax2.set_yticks(np.arange(-1, 1.1, 0.5))
    
    mn = pd.to_timedelta("100 days")
    ax.set_xlim(t[0]-mn, t[-1]+mn)
    
    tt = pd.date_range(f'{sy}', periods=10, freq='2AS-JAN')
    ax.set_xticks(tt)
    ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m'))
    ax.tick_params(labelsize=20)
    ax.tick_params(which="major", bottom=1, left=1, length=8)
    ax2.tick_params(labelsize=20)
    ax2.tick_params(which="major", bottom=1, left=0, length=8)
    
    #### 添加图例，本质上line1为list
    line1, label1 = ax.get_legend_handles_labels()
    line2, label2 = ax2.get_legend_handles_labels()
    lines = line1+line2
    labels = label1+label2
    
    
    return ax, lines, labels


def plot2(data1, data2, df, title):
    print(np.nanmax(data1), np.nanmin(data1))
    print(np.nanmax(data2), np.nanmin(data2))
    
    fig = plt.figure(1, figsize=(22, 12), dpi=500)
    
    fig.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.93, wspace=None, hspace=0.12)
    axs = fig.subplots(2, 1, sharey=True)
    
    a, b = np.zeros(480), np.zeros(480)
    a[0:12], a[12:456], a[456:] = np.nan, data1, np.nan
    b[0:12], b[12:456], b[456:] = np.nan, data2, np.nan
    
    data1 = a
    data2 = b
    
    axs[0], lines, labels = subplot2(data1[:240], data2[:240], axs[0], 1981, df)
    axs[1], lines, labels = subplot2(data1[240:], data2[240:], axs[1], 2001, df)
    
    fig.legend(lines, labels, loc = 'upper right', bbox_to_anchor=(0.95, 1), fontsize=20)
    plt.xlabel("years", fontsize=25)
    plt.suptitle(f'{title}', fontsize=30)
    plt.savefig(rf'E:/GLASS-GPP/JPG MG/{title}.jpg', 
                bbox_inches='tight')
    plt.show()
    
#%%
df = pd.read_excel("E:/SPEI_base/python MG/MG Grassland干旱统计.xlsx", sheet_name='Grassland')

plot(gpp_MG, gpp_a_MG, df, f"MG Grassland GPP")

#%%
plot2(wue_MG, wue_a_MG, df, f"MG Grassland WUE")


