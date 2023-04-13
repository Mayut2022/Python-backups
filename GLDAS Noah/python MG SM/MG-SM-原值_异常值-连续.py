# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 19:58:59 2022

@author: MaYutong
"""

import datetime as dt
import netCDF4 as nc
from matplotlib import font_manager
import matplotlib.dates as mdate
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
import xarray as xr

plt.rcParams['font.sans-serif']='times new roman'
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
    global sm, lat2, lon2
    inpath = rf"E:/GLDAS Noah/DATA_RG/SM_81_20_ORINGINAL_SPEI0.5x0.5.nc"
    with nc.Dataset(inpath, mode='r') as f:
        
        #print(f.variables.keys())
        
        sm = (f.variables['sm'][:, :, :])
        lat2 = (f.variables['lat'][:])
        lon2 = (f.variables['lon'][:])
        
#%%
def read_nc3():
    global sm_a
    inpath = rf"E:/GLDAS Noah/DATA_RG/SM_Anom_81_20_ORINGINAL_SPEI0.5x0.5.nc"
    with nc.Dataset(inpath, mode='r') as f:
        
        #print(f.variables.keys())
        
        sm_a = (f.variables['sm'][:, :, :])

        
# %%
def MG(data):
    t = np.arange(1, 481, 1)
    l = np.arange(1, 5, 1)
    spei_global = xr.DataArray(data, dims=['t', 'l', 'y', 'x'], coords=[
                               t, l, lat2, lon2])  # 原SPEI-base数据
    spei_rg1 = spei_global.loc[:, :, 40:55, 100:125]
    spei_rg1 = np.array(spei_rg1)
    return spei_rg1

#%% mask数组
def mask(x, data):
    lcc2 = np.array(lcc)
    
    np.place(lcc2, lcc2!=x, 1)
    np.place(lcc2, lcc2==x, 0)
    
    spei_ma = np.empty((480, 4, 30, 50))
    spei_ma = np.ma.array(spei_ma)
    
    for i in range(480):
        for l in range(4):
            a = data[i, l, :, :]
            a = ma.masked_array(a, mask=lcc2)
            spei_ma[i, l, :, :] = a

    spei_ma_ave = np.nanmean(spei_ma, axis=(2, 3))
    
    return spei_ma_ave


#%%
def plot(data1, data2, title):
    fig, axs = plt.subplots(4, 1, figsize=(20, 13), dpi=150, sharex=True)
    fig.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.93, wspace=0.05, hspace=0.15)
    
    df = pd.read_excel("E:/SPEI_base/python MG/MG Grassland干旱统计.xlsx", 
                       sheet_name='Grassland')
    t = pd.date_range(f'1981', periods=480, freq="MS")
    
    
        
    for i in range(4):
        for y1, m1, y2, m2 in zip(df["SY"], df["SM"], df["EY"], df["EM"]):
            d1 = dt.datetime(y1, m1, 1, 0, 0, 0)
            d2 = dt.datetime(y2, m2, 1, 0, 0, 0)+dt.timedelta(days=35)
            axs[i].fill_between([d1, d2], 280, -2.8, facecolor='dodgerblue', alpha=0.4) ## 貌似只要是时间格式的x都行
            
        
        axs[i].scatter(t, data1[:, i], c='k', s=2) #
        lin2 = axs[i].plot(t, data1[:, i], c='k', label="SM", zorder=2) #
        
        if i+1==1:
            axs[i].set_ylim(0, 40)
            axs[i].set_yticks(np.arange(0, 41, 10))
            axs[i].set_title(f"Layer1 (0-10 cm)", fontsize=20, loc="left") #
        elif i+1==2:
            axs[i].set_ylim(0, 80)
            axs[i].set_yticks(np.arange(0, 81, 20))
            axs[i].set_title(f"Layer2 (10-40 cm)", fontsize=20, loc="left") #
        elif i+1==3:
            axs[i].set_ylim(40, 120)
            axs[i].set_yticks(np.arange(40, 121, 20))
            axs[i].set_title(f"Layer3 (40-100 cm)", fontsize=20, loc="left") #
        elif i+1==4:
            axs[i].set_ylim(160, 240)
            axs[i].set_yticks(np.arange(160, 241, 20))
            axs[i].set_title(f"Layer4 (100-200 cm)", fontsize=20, loc="left") #
            
        mn = pd.to_timedelta("100 days")
        axs[i].set_xlim(t[0]-mn, t[-1]+mn)
        
        tt = pd.date_range(f'1981', periods=11, freq='4AS-JAN')
        axs[i].set_xticks(tt)
        axs[i].xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m'))
        axs[i].tick_params(labelsize=20)
        
        ax2 = axs[i].twinx()
        lin1 = ax2.bar(t, data2[:, i], color='brown', label="SM Anomaly", alpha=0.8, width=25) #
        if i+1==1 or i+1==4:
            ax2.set_ylim(-20, 20)
            ax2.set_yticks(np.arange(-20, 21, 10))
        else:
            ax2.set_ylim(-40, 40)
            ax2.set_yticks(np.arange(-40, 41, 20))
        ax2.tick_params(axis='y', labelsize=20, colors="brown")
        ax2.spines["right"].set_color("brown")
        
    
    #合并图例
    
    #plt.set_ylabel("Soil Moisture (kg m-2 month mean)", fontsize=15)
    plt.suptitle(f"{title}", fontsize=25)
    plt.savefig(rf'E:/GLDAS Noah/JPG_MG/{title}.jpg', 
                bbox_inches='tight')
    plt.show()
        
#%%
read_nc()
read_nc2()
read_nc3()

sm_MG = MG(sm)
sm_a_MG = MG(sm_a)

sm_MG_ave = mask(130, sm_MG)
sm_a_MG_ave = mask(130, sm_a_MG)

#%%
print(np.nanmax(sm_MG_ave), np.nanmin(sm_MG_ave))
print(np.nanmax(sm_a_MG_ave), np.nanmin(sm_a_MG_ave), "\n")

data1 = sm_MG_ave
data2 = sm_a_MG_ave

plot(data1, data2,  f"MG Grassland SM All Layer")
    
