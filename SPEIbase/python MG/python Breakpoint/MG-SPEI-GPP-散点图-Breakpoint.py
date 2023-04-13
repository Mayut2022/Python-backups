# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 13:30:14 2022

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
def sns_plot(df_g, df_g_break, title):
    
    data_x = df_g.iloc[:, 3:21]
    data_y = df_g.iloc[:, 21:]
    '''
    print(data_x.max(), data_x.min())
    print(data_y.max(), data_y.min())
    '''
    df_g_break["quantile"] = df_g_break.index
    df_g_break.index = np.arange(350)
    
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
                
                x2 = df_g_break[f"GPP{ind}"]
                y2 = df_g_break[f"SPEI{ind}"]
                
                sns.scatterplot(x=y, y=x, ax=axes[i, j], **kws)
                sns.lineplot(x=y2, y=x2, hue="quantile", data=df_g_break, ax=axes[i, j])
                #sns.regplot(x=y, y=x, ax=axes[i, j], scatter_kws=kws)
                
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
    plt.savefig(rf'E:/SPEI_base/python MG/JPG/响应时间分段/{title}.jpg', 
                bbox_inches='tight')
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

#%% reshape all
def reshape(data, name):
    data = np.array(data)
    sp = data.shape
    data = data.reshape(sp[0]*sp[1], 1)
    data = np.squeeze(data)
    data = pd.Series(data, name=name)
    return data

#%% Breakpoint 验证
def breakpoint(df):
    yr_len = (df.shape[1])/2
    for yr in range(1, int(yr_len+1)):
        a = df[[f"GPP{yr}", f"SPEI{yr}"]]
        bin = np.arange(-2.5, 2.51, 0.1)
        
        b = a.sort_values(by=[f"SPEI{yr}"])
        for i in range(len(bin)-1):
            c = b[np.logical_and(b[f"SPEI{yr}"]>bin[i], b[f"SPEI{yr}"]<bin[i+1])]
            d = c.quantile(q=[0.95, 0.9, 0.75, 0.5, 0.25, 0.1, 0.05])
            if i==0:
                d_all = d
            else:
                d_all = pd.concat([d_all, d], axis=0)
        
        if yr==1:
            d_all_yr = d_all
        else:
            d_all_yr = pd.concat([d_all_yr, d_all], axis=1)
            
    return d_all_yr

def breakpoint2(df):
    bin = np.arange(-2.5, 2.51, 0.1)
    b = df.sort_values(by=[f"SPEI"])
    for i in range(len(bin)-1):
        c = b[np.logical_and(b[f"SPEI"]>bin[i], b[f"SPEI"]<bin[i+1])]
        d = c.quantile(q=[0.95, 0.9, 0.75, 0.5, 0.25, 0.1, 0.05])
        if i==0:
            d_all = d
        else:
            d_all = pd.concat([d_all, d], axis=0)
            
    return d_all
        
#%% 逐年
'''
Rmax1 = [10, 10, 11]
scale = [str(x).zfill(2) for x in Rmax1]

for mn, s in zip(month[2:3], scale[:1]):
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
    
    df1_g_break = breakpoint(df1_g)
    
    sns_plot(df1_g, df1_g_break, title=f"GPP-SPEI Scatter BreakPoint 82-99 {mn_str[mn-1]}")
    #sns_plot(df2_g, title=f"GPP-SPEI Scatter 00-18 {mn_str[mn-1]}")
    '''


#%%
def sns_plot_all(df_g_break, title):
    df_g_break["quantile"] = df_g_break.index
    df_g_break.index = np.arange(350)
    
    x2 = df_g_break[f"GPP"]
    y2 = df_g_break[f"SPEI"]
    
    x = df_g_all[f"GPP"]
    y = df_g_all[f"SPEI"]
    
    ##########
    fig, axes = plt.subplots(1, 1, figsize=(10, 10), dpi=500, sharey=True, sharex=True)
    fig.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.93, wspace=0.1, hspace=0.15)

    sns.set_theme(style="ticks")
    kws = dict(color="orange", alpha=0.2, edgecolor="w")
    sns.scatterplot(x=y, y=x, ax=axes, **kws)
    sns.lineplot(x=y2, y=x2, hue="quantile", data=df_g_break, linewidth=3, legend="full", ax=axes)
    axes.tick_params(labelsize=20)
    axes.set_xlim(-2.8, 2.8)
    axes.set_xticks(np.arange(-2.5, 2.6, 0.5))
    axes.set_ylim(-10, 280)
    axes.set_yticks(np.arange(0, 251, 50))
    axes.set_xlabel("SPEI", fontsize=20)
    axes.set_ylabel("GPP", fontsize=20)
    
    plt.legend(loc="upper left", fontsize=20)
    plt.suptitle(f'{title}', fontsize=30)
    plt.savefig(rf'E:/SPEI_base/python MG/JPG/响应时间分段/{title}.jpg', 
                bbox_inches='tight')
    plt.show()
    
#%% 全部的点
'''
Rmax1 = [10, 2, 3]
scale = [str(x).zfill(2) for x in Rmax1]

for mn, s in zip(month[2:5], scale[:]):
    print(f"month {mn} is in programming!")
    print(f"scale spei{s} is in programming!")
    df = origin_df()
    spei_s = read_nc2(s)
    gpp_mn = gpp_MG[:, mn-1, :, :]
    spei_mn = spei_s[mn-1, 1:38, :,:]
    df1 = pd.concat([df, reshape1(gpp_mn)], axis=1)
    df2 = pd.concat([df, reshape1(spei_mn)], axis=1)
    
    print(f"Next is sns plot-----")
    df_g = df1[df1["LCC"]==130].iloc[:, 3:]
    df_g = reshape(df_g, "GPP")
    df2_g = df2[df2["LCC"]==130].iloc[:, 3:]
    df2_g = reshape(df2_g, "SPEI")
    
    df_g_all = pd.concat([df_g, df2_g], axis=1)
    df_g_break = breakpoint2(df_g_all)
    sns_plot_all(df_g_break, title=f"GPP-SPEI BreakPoint {mn_str[mn-1]}")

'''

#%%
def sns_plot2(df_g, df_g_break, title):
    global line1
    data_x = df_g.iloc[:, 0::2]
    data_y = df_g.iloc[:, 1::2]
    
    #print(data_x.max(), data_x.min())
    #print(data_y.max(), data_y.min())
    
    df_g_break["quantile"] = df_g_break.index
    df_g_break.index = np.arange(350)
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 12), dpi=500, sharey=True, sharex=True)
    fig.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.93, wspace=0.1, hspace=0.15)
    
    sns.set_theme(style="ticks")
    
    title_str = ["82-87", "88-93", "94-99", "00-05", "06-11", "12-18"]
    ind=1
    for i in range(2):
        for j in range(3):
            kws = dict(color="orange", alpha=0.5, edgecolor="w")
    
            if data_x.shape[1]>=ind:
                x = df_g[f"GPP{ind}"]
                y = df_g[f"SPEI{ind}"]
                
                x2 = df_g_break[f"GPP{ind}"]
                y2 = df_g_break[f"SPEI{ind}"]
                
                sns.scatterplot(x=y, y=x, ax=axes[i, j], **kws)
                sns.lineplot(x=y2, y=x2, hue="quantile", 
                             data=df_g_break, legend=False, ax=axes[i, j])
                
                axes[i, j].set_title(f"{title_str[ind-1]}", fontsize=15, loc="left")
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
        
     #### 添加图例，本质上line1为list
    line1, label1 = axes[0, 0].get_legend_handles_labels()
    fig.legend(line1, label1, loc = 'upper right', bbox_to_anchor=(0.95, 1), fontsize=20) 
    
    plt.suptitle(f'{title}', fontsize=30)
    plt.savefig(rf'E:/SPEI_base/python MG/JPG/响应时间分段/{title}.jpg', 
                bbox_inches='tight')
    plt.show()
    
#%% 每五年，看变化
Rmax1 = [10, 2, 3]
scale = [str(x).zfill(2) for x in Rmax1]
ind = list(range(0, 31, 6))
ind.append(37)

for mn, s in zip(month[2:5], scale[:]):
    print(f"month {mn} is in programming!")
    print(f"scale spei{s} is in programming!")
    df = origin_df()
    spei_s = read_nc2(s)
    gpp_mn = gpp_MG[:, mn-1, :, :]
    spei_mn = spei_s[mn-1, 1:38, :,:]
    for i in range(6):
        df1 = pd.concat([df, reshape1(gpp_mn[ind[i]:ind[i+1], :, :])], axis=1)
        df2 = pd.concat([df, reshape2(spei_mn[ind[i]:ind[i+1], :, :])], axis=1)
        df1_g = df1[df1["LCC"]==130].iloc[:, 3:]
        df1_g = reshape(df1_g, "GPP")
        df2_g = df2[df2["LCC"]==130].iloc[:, 3:]
        df2_g = reshape(df2_g, "SPEI")
        df_g = pd.concat([df1_g, df2_g], axis=1)
        
        df_g_break = breakpoint2(df_g)
        
        df_g.columns = [f"GPP{i+1}", f"SPEI{i+1}"]
        df_g_break.columns = [f"GPP{i+1}", f"SPEI{i+1}"]
        if i==0:
            df_g_all, df_g_break_all = df_g, df_g_break
        else:
            df_g_all = pd.concat([df_g_all, df_g], axis=1)
            df_g_break_all = pd.concat([df_g_break_all, df_g_break], axis=1)
            
    sns_plot2(df_g_all, df_g_break_all, title=f"GPP-SPEI BreakPoint Change {mn_str[mn-1]}")

