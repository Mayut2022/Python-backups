# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 12:07:18 2022

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
def read_nc6():
    global e, latg, long
    inpath = r"E:/Gleamv3.6a/v3.6a/global/E_1980-2021_GLEAM_v3.6a_MO_SPEI_0.5X0.5.nc"
    with nc.Dataset(inpath) as f:
        #print(f.variables.keys())
        
        latg = (f.variables['lat'][:])
        long = (f.variables['lon'][:])
        e = (f.variables['E'][:])
 
#%% 提取区域
def pre_xarray(band1):
    t = np.arange(40)
    mn = np.arange(1, 13, 1)
    pre=xr.DataArray(band1,dims=['mn', 't', 'y','x'],coords=[mn, t, latg, long])
    pre_MG = pre.loc[:, :, 40:55, 100:125]
    lat_MG = pre_MG.y
    lon_MG = pre_MG.x
    pre_MG = np.array(pre_MG)
    return pre_MG, lat_MG, lon_MG

#%% 480 -> 月 年
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

#%% mask数组
def mask2(x, data):
    lcc2 = np.array(lcc)
    
    np.place(lcc2, lcc2!=x, 1)
    np.place(lcc2, lcc2==x, 0)
    
    spei_ma = np.empty((12, 40, 30, 50))
    spei_ma = np.ma.array(spei_ma)
    
    for i in range(12):
        for l in range(40):
            a = data[i, l, :, :]
            a = ma.masked_array(a, mask=lcc2)
            spei_ma[i, l, :, :] = a

    spei_ma_ave = np.nanmean(spei_ma, axis=(2, 3))
    
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
    data_all = data_all.reshape(37, 12)
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
    data_all = data_all.reshape(37, 12)   
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
    data_all = data_all.reshape(37, 12)    
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
    data_all = data_all.reshape(37, 12)   
    return data_all

#%% GPP
gpp_MG = exact_data1()
gpp_sum = gpp_MG.sum(axis=1)
#gpp_a_MG = exact_data2()
#wue_MG = exact_data3()
#wue_a_MG = exact_data4()
read_nc6()
e_mn = mn_yr(e)
e_RG, _, _ = pre_xarray(e_mn)

_ = mask2(130, e_RG)
e_sum = _.sum(axis=0)

wue_sum = gpp_sum/e_sum[2:-1]

#%%
def plot(data1, data2, data3, title):
    print(np.nanmax(data1), np.nanmin(data1))
    print(np.nanmax(data2), np.nanmin(data2))
    print(np.nanmax(data3), np.nanmin(data3))
    
    fig, axs = plt.subplots(3, 1, figsize=(20, 13), dpi=150, sharex=True)
    fig.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.93, wspace=0.05, hspace=0.15)
    
    df = pd.read_excel("E:/SPEI_base/python MG/MG Grassland干旱统计.xlsx", 
                       sheet_name='Grassland')
    
    t = pd.date_range(f'1981', periods=40, freq="YS")
    t2 = pd.date_range(f'1982', periods=37, freq="YS")
    
    for i in range(3):
        for y1, m1, y2, m2 in zip(df["SY"], df["SM"], df["EY"], df["EM"]):
            d1 = dt.datetime(y1, m1, 1, 0, 0, 0)
            d2 = dt.datetime(y2, m2, 1, 0, 0, 0)+dt.timedelta(days=35)
            axs[i].fill_between([d1, d2], 600, -2.8, facecolor='dodgerblue', alpha=0.4) ## 貌似只要是时间格式的x都行
        
        #axs[i].fill_between([t[20], t[-1]], 600, -2.8, facecolor='red', alpha=0.2)    
        
        exec(f'axs[i].scatter(t2, data{i+1}, c="k", s=20)') #
        exec(f'axs[i].plot(t2, data{i+1}, c="k", zorder=2, linewidth=2)') #
        exec(f'axs[i].axhline(y=data{i+1}.mean(), c="r", linestyle="--")')
        
        if i+1==1:
            axs[i].set_ylim(390, 600)
            axs[i].set_yticks(np.arange(400, 601, 50))
            axs[i].set_title(f"GPP (gC m-2 month-1)(GLASS)", fontsize=20, loc="left") #
        elif i+1==2:
            axs[i].set_ylim(260, 380)
            axs[i].set_yticks(np.arange(260, 381, 20))
            axs[i].set_title(f"ET (mm month-1)(GLEAM)", fontsize=20, loc="left") #
        elif i+1==3:
            axs[i].set_ylim(1, 2)
            axs[i].set_yticks(np.arange(1, 2.1, 0.2))
            axs[i].set_title(f"WUE (gC m-2 mm-1)", fontsize=20, loc="left") #
            
        mn = pd.to_timedelta("100 days")
        axs[i].set_xlim(t[0]-mn, t[-1]+mn)
        
        tt = pd.date_range(f'1982', periods=19, freq='2AS-JAN')
        axs[i].set_xticks(tt)
        axs[i].xaxis.set_major_formatter(mdate.DateFormatter('%Y'))
        axs[i].tick_params(labelsize=15)

    
    #plt.set_ylabel("Soil Moisture (kg m-2 month mean)", fontsize=15)
    plt.suptitle(f"{title}", fontsize=25)
    plt.savefig(rf'E:/GLASS-GPP/JPG MG/{title}.jpg', 
                bbox_inches='tight')
    plt.show()

#%%
plot(gpp_sum, e_sum[2:-1], wue_sum, title="MG Grass GPP ET WUE Year")