# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 20:22:05 2022

@author: MaYutong
"""

import netCDF4 as nc

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
import xarray as xr
import xlsxwriter

import seaborn as sns

from scipy.stats import linregress
from scipy.stats import pearsonr


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
def read_nc(inpath):
    global gpp, lat2, lon2
    with nc.Dataset(inpath) as f:
        #print(f.variables.keys())
        
        gpp = (f.variables['GPP'][:])
        lat2 = (f.variables['lat'][:])
        lon2 = (f.variables['lon'][:])
        
    return gpp

def sif_xarray2(band1):
    mn = np.arange(12)
    sif=xr.DataArray(band1, dims=['mn', 'y','x'],coords=[mn, lat2, lon2])
    
    sif_MG = sif.loc[:, 40:55, 100:125]
    return np.array(sif_MG)


def exact_data1():
    for yr in range(1982, 2019):
        inpath =  rf"E:/GLASS-GPP/Month RG/GLASS_GPP_RG_SPEI_0.5X0.5_{yr}.nc"
        data = read_nc(inpath)
        
        data_MG = sif_xarray2(data)
        data_MG = data_MG.reshape(1, 12, 30, 50)
        
        if yr == 1982:
            data_all = data_MG
        else:
            data_all = np.vstack((data_all, data_MG))
        
    return data_all

#%%
def read_nc2(scale):
    inpath = (rf"E:/SPEI_base/data/spei{scale}.nc")
    with nc.Dataset(inpath) as f:
        #print(f.variables.keys())
        
        spei = (f.variables['spei'][960:])
        time = (f.variables['time'][960:])
        t2 = nc.num2date(time, 'days since 1900-01-01 00:00:0.0').data
        lat_g = (f.variables['lat'][:])
        lon_g = (f.variables['lon'][:])
        
    def sif_xarray(band1):
        mn = np.arange(12)
        yr = np.arange(40)
        sif=xr.DataArray(band1, dims=['mn', 'yr', 'y','x'],coords=[mn, yr, lat_g, lon_g])
        
        sif_MG = sif.loc[:, :, 40:55, 100:125]
        return np.array(sif_MG)
    
    spei_mn = mn_yr(spei)
    spei_mn_MG = sif_xarray(spei_mn)
    
    return spei_mn_MG

#%%
def read_nc3():
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
def sns_plot(df_g, title):
    data_x = df_g.iloc[:, 3:21]
    data_y = df_g.iloc[:, 21:]
    print(data_x.max(), data_x.min())
    print(data_y.max(), data_y.min())
    
    fig, axes = plt.subplots(4, 5, figsize=(16, 13), dpi=500, sharey=True, sharex=True)
    fig.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.93, wspace=0.1, hspace=0.15)
    
    sns.set_theme(style="ticks")
    
    ind=1
    for i in range(4):
        for j in range(5):
            kws = dict(color="orange", alpha=1, edgecolor="w")
            
            
            if data_x.shape[1]>=ind:
                x = df_g[f"GPP{ind}"]
                y = df_g[f"SPEI{ind}"]
                
                #sns.scatterplot(x=y, y=x, ax=axes[i, j], **kws)
                sns.regplot(x=y, y=x, ax=axes[i, j], scatter_kws=kws)
                
                axes[i, j].text(-2.6, 270, f"{ind+1981}", fontsize=15)
                axes[i, j].tick_params(labelsize=15)
                axes[i, j].set_xlim(-2.8, 2.8)
                axes[i, j].set_xticks(np.arange(-2, 2.1, 1))
                axes[i, j].set_ylim(-10, 300)
                axes[i, j].set_yticks(np.arange(0, 301, 50))
                
                #axes[i, j].set_xlabel("SPEI", fontsize=15)
                #axes[i, j].set_ylabel("GPP", fontsize=15)
                axes[i, j].set_xlabel(None)
                axes[i, j].set_ylabel(None)
                
            
                ind += 1
            else:
                axes[i, j].set_visible(False)
        
    plt.suptitle(f'{title}', fontsize=30)
    # plt.savefig(rf'E:/SPEI_base/python MG/JPG/响应时间分段/{title}.jpg', 
    #             bbox_inches='tight')
    plt.show()
    
#%%
gpp_MG = exact_data1()
read_nc3()

def origin_df():
    lON, lAT = np.meshgrid(lon, lat)
    lcc_re, lon_re, lat_re = lcc.reshape(30*50), lON.reshape(30*50), lAT.reshape(30*50)
    dic = dict(LCC=lcc_re, LAT=lat_re, LON=lon_re)
    df = pd.DataFrame(dic)
    
    return df

df = origin_df()

#%%
month = np.arange(4, 11, 1) ## Apr-Oct 生长季
DF = pd.read_excel('E:/ERA5/每月天数.xlsx')
mn_str = DF['月份']
    
    
#%% scatter 
def reshape1(data):
    sp = data.shape
    data = pd.DataFrame(data.reshape(sp[0], sp[1]*sp[2])).T
    data.columns = [f"GPP{yr}" for yr in range(1, sp[0]+1)]
    return data

def reshape2(data):
    sp = data.shape
    data = pd.DataFrame(data.reshape(sp[0], sp[1]*sp[2])).T
    data.columns = [f"SPEI{yr}" for yr in range(1, sp[0]+1)]
    return data

#%%
Rmax1 = [10, 10, 11]
scale = [str(x).zfill(2) for x in Rmax1]

for mn, s in zip(month[2:5], scale):
    print(f"month {mn} is in programming!")
    print(f"scale spei{s} is in programming!")
    gpp_mn1 = gpp_MG[:18, mn-1, :, :]
    gpp_mn2 = gpp_MG[18:, mn-1, :, :]
    df = origin_df()
    df1 = pd.concat([df, reshape1(gpp_mn1)], axis=1)
    df2 = pd.concat([df, reshape1(gpp_mn2)], axis=1)
    spei_s = read_nc2(s)
    spei_mn1 = spei_s[mn-1, 1:19, :,:]
    spei_mn2 = spei_s[mn-1, 19:38, :,:]
    df1 = pd.concat([df1, reshape2(spei_mn1)], axis=1)
    df2 = pd.concat([df2, reshape2(spei_mn2)], axis=1)
        
    print(f"Next is sns plot-----")
    df1_g = df1[df1["LCC"]==130]
    df2_g = df2[df2["LCC"]==130]
    sns_plot(df1_g, title=f"GPP-SPEI Scatter 82-99 {mn_str[mn-1]}")
    sns_plot(df2_g, title=f"GPP-SPEI Scatter 00-18 {mn_str[mn-1]}")
    
#%% reshape 验证
'''
a = []
for i in range(18):
    a.append(lAT*i)
    
a = np.array(a)

b = reshape1(a)
'''