# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 17:16:08 2022

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


#%%
def read_nc():
    inpath = r"E:/GIMMS_NDVI/data_RG/NDVI_82_15_RG_SPEI0.5x0.5.nc"
    with nc.Dataset(inpath) as f:
        #print(f.variables.keys())
        
        ndvi = (f.variables['ndvi'][:])
        lat2 = (f.variables['lat'][:])
        lon2 = (f.variables['lon'][:])
        
    def ndvi_xarray(band1):
        t = np.arange(34)
        mn = np.arange(1, 13, 1)
        ndvi=xr.DataArray(band1,dims=['t','mn', 'y','x'],coords=[t, mn, lat2, lon2])
        ndvi_IM = ndvi.loc[:, :, 40:55, 100:125]
        lat_IM = ndvi_IM.y
        lon_IM = ndvi_IM.x
        ndvi_IM = np.array(ndvi_IM)
        return ndvi_IM, lat_IM, lon_IM
    
    ndvi_MG, _, _ = ndvi_xarray(ndvi)
    return ndvi_MG

        

#%%
def read_nc2():
    global lat, lon
    # global a1, a2, o1, o2
    inpath = (r"E:\ESA\ESA_LUC_2000_SPEI03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        # print(f.variables['lccs_class'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        lcc = (f.variables['lcc'][:])
        
    return lcc
#%%
def read_nc3():
    inpath = r"E:/CRU/Q_DATA_CRU-GLEAM/PRE_global_81_20.nc"
    with nc.Dataset(inpath) as f:
        #print(f.variables.keys())
        
        lat_g = (f.variables['lat'][:])
        lon_g = (f.variables['lon'][:])
        pre = (f.variables['pre'][:])
        #480(输入data) -> 月 年
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
        
        
        pre_mn = mn_yr(pre)
        def sif_xarray(band1):
            mn = np.arange(12)
            yr = np.arange(40)
            sif=xr.DataArray(band1, dims=['mn', 'yr', 'y','x'],coords=[mn, yr, lat_g, lon_g])
            
            sif_MG = sif.loc[:, :, 40:55, 100:125]
            return np.array(sif_MG)
        
    pre_mn_MG = sif_xarray(pre_mn)
    return pre_mn_MG


#%% scatter 
def reshape1(data):
    sp = data.shape
    data = pd.DataFrame(data.reshape(sp[0], sp[1]*sp[2])).T
    data.columns = [f"NDVI{yr}" for yr in range(1, sp[0]+1)]
    return data

def reshape2(data):
    sp = data.shape
    data = pd.DataFrame(data.reshape(sp[0], sp[1]*sp[2])).T
    data.columns = [f"PRE{yr}" for yr in range(1, sp[0]+1)]
    return data

#%% reshape all
def reshape(data, name):
    data = np.array(data)
    sp = data.shape
    data = data.reshape(sp[0]*sp[1], 1)
    data = np.squeeze(data)
    data = pd.Series(data, name=name)
    return data

#%%
def breakpoint(df):
    yr_len = (df.shape[1]-3)/2
    for yr in range(1, int(yr_len+1)):
        a = df[[f"NDVI{yr}", f"PRE{yr}"]]
        bin = np.arange(80, 781, 20)
        
        b = a.sort_values(by=[f"PRE{yr}"])
        for i in range(len(bin)-1):
            c = b[np.logical_and(b[f"PRE{yr}"]>bin[i], b[f"PRE{yr}"]<bin[i+1])]
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
    bin = np.arange(80, 781, 20)
    b = df.sort_values(by=[f"PRE"])
    for i in range(len(bin)-1):
        c = b[np.logical_and(b[f"PRE"]>bin[i], b[f"PRE"]<bin[i+1])]
        d = c.quantile(q=[0.95, 0.9, 0.75, 0.5, 0.25, 0.1, 0.05])
        if i==0:
            d_all = d
        else:
            d_all = pd.concat([d_all, d], axis=0)
            
    return d_all


#%%
def sns_plot_all(df_g_break, title):
    df_g_break["quantile"] = df_g_break.index
    df_g_break.index = np.arange(245)
    
    x2 = df_g_break[f"NDVI"]
    y2 = df_g_break[f"PRE"]
    
    x = df_g_all[f"NDVI"]
    y = df_g_all[f"PRE"]
    
    #########
    q = [1, 0.95, 0.9, 0.75, 0.5, 0.25, 0.1, 0.05]
    print(x2.quantile(q=q))
    print(x.quantile(q=q))
    
    ##########
    fig, axes = plt.subplots(1, 1, figsize=(12, 12), dpi=500, sharey=True, sharex=True)
    fig.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.93, wspace=0.1, hspace=0.15)

    sns.set_theme(style="ticks")
    kws = dict(color="green", alpha=0.2, edgecolor="w")
    sns.scatterplot(x=y, y=x, ax=axes, **kws)
    sns.lineplot(x=y2, y=x2, hue="quantile", data=df_g_break, 
                 linewidth=3, legend="full", ax=axes)
    axes.tick_params(labelsize=20)
    axes.set_xlim(0, 800)
    axes.set_xticks(np.arange(0, 801, 100))
    axes.set_ylim(0, 0.7)
    axes.set_yticks(np.arange(0, 0.71, 0.1))
    axes.set_xlabel("PRE (mm/year)", fontsize=20)
    axes.set_ylabel("NDVI (GS MEAN)", fontsize=20)
    
    plt.legend(loc="upper left", fontsize=20)
    plt.suptitle(f'{title}', fontsize=30)
    plt.savefig(rf'E:/CRU/JPG_MG/{title}.jpg', 
                bbox_inches='tight')
    plt.show()


def sns_plot(df_g, df_g_break, title):
    
    data_x = df_g.iloc[:, 3::2]
    data_y = df_g.iloc[:, 4::2]
    '''
    print(data_x.max(), data_x.min())
    print(data_y.max(), data_y.min())
    '''
    df_g_break["quantile"] = df_g_break.index
    df_g_break.index = np.arange(245)
    
    fig, axes = plt.subplots(5, 6, figsize=(15, 12), dpi=500, sharex=True, sharey=True)
    fig.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.93, wspace=0.2, hspace=0.15)
    
    sns.set_theme(style="ticks")
    
    ind=1
    for i in range(5):
        for j in range(6):
            kws = dict(color="green", alpha=0.3, s=30, edgecolor="w")

            if data_x.shape[1]>ind:
                x = df_g[f"NDVI{ind}"]
                y = df_g[f"PRE{ind}"]
                
                x2 = df_g_break[f"NDVI{ind}"]
                y2 = df_g_break[f"PRE{ind}"]
                
                sns.scatterplot(x=y, y=x, ax=axes[i, j], **kws)
                sns.lineplot(x=y2, y=x2, hue="quantile", data=df_g_break, 
                             legend=None, ax=axes[i, j])
                
                axes[i, j].text(20, 0.6, f"{ind+1983}", fontsize=15)
                axes[i, j].tick_params(labelsize=15)
                axes[i, j].set_xlim(0, 800)
                axes[i, j].set_xticks(np.arange(0, 801, 200))
                axes[i, j].set_ylim(0, 0.7)
                axes[i, j].set_yticks(np.arange(0, 0.71, 0.1))
                ind += 1
            
            ##### 最后一个子图用于调整图例位置
            elif ind==30:
                sns.scatterplot(x=y, y=x, ax=axes[i, j], **kws)
                g = sns.lineplot(x=y2, y=x2, hue="quantile", data=df_g_break, 
                             legend="full", ax=axes[i, j])
                g.legend(bbox_to_anchor=(0.8, 1), ncol=2, fontsize=15, 
                         title="Quantile", title_fontsize=15) 
                axes[i, j].text(20, 0.6, f"{ind+1983}", fontsize=15)
                axes[i, j].tick_params(labelsize=15)
                axes[i, j].set_xlim(0, 800)
                axes[i, j].set_xticks(np.arange(0, 801, 200))
                axes[i, j].set_ylim(0, 0.7)
                axes[i, j].set_yticks(np.arange(0, 0.71, 0.2))
                ind += 1
                
            else:
                axes[i, j].set_visible(False)
                
            ########### 修饰label
            if i==4:
                axes[i, j].set_xlabel("PRE", fontsize=15)
            else:
                axes[i, j].set_xlabel(None)
                
            if j==0:
                axes[i, j].set_ylabel("GPP", fontsize=15)
            else:
                axes[i, j].set_ylabel(None)
    
    plt.suptitle(f'{title}', fontsize=30)
    plt.savefig(rf'E:/CRU/JPG_MG/{title}.jpg', 
                bbox_inches='tight')
    plt.show()


#%% 读取数据
ndvi = read_nc()
lcc = read_nc2()
pre = read_nc3()
pre = pre[:, 1:35, :, :]

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
    
#%% 逐年相关性
'''
ndvi_yr = ndvi[:, 3:10, :, :].mean(axis=1)    
pre_yr = pre.sum(axis=0)

df1 = pd.concat([df, reshape1(ndvi_yr), reshape2(pre_yr)], axis=1)
df_g = df1[df1["LCC"]==130]
df1_g = reshape(df_g.iloc[:, 3:37], "NDVI")
df2_g = reshape(df_g.iloc[:, 37:], "PRE")
df_g_all = pd.concat([df1_g, df2_g], axis=1)
df_g_break = breakpoint2(df_g_all)
sns_plot_all(df_g_break, title=f"NDVI-PRE BreakPoint Annual")

'''
#%% 五年滑动平均
for i in range(30):
    ndvi_yr = ndvi[i:5+i, 3:10, :, :].mean(axis=1).mean(axis=0)    
    pre_yr = pre[:, i:5+i, :, :].sum(axis=0).mean(axis=0) 
    if i==0:
        df1 = pd.concat([df, reshape(ndvi_yr, f"NDVI{i+1}"), reshape(pre_yr, f"PRE{i+1}")], axis=1)
    else:
        df1 = pd.concat([df1, reshape(ndvi_yr, f"NDVI{i+1}"), reshape(pre_yr, f"PRE{i+1}")], axis=1)
    df_g = df1[df1["LCC"]==130]
    
    df_g_break = breakpoint(df_g)

sns_plot(df_g, df_g_break, "NDVI-PRE Breakpoint 5-year running mean")
