# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 11:50:19 2022

@author: MaYutong
"""

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import numpy.ma as ma

import pandas as pd
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from scipy.stats import chi2_contingency
from sklearn.linear_model import LinearRegression

import xarray as xr

#%%
def read_nc0():
    global lcc
    # global a1, a2, o1, o2
    inpath = (r"E:\ESA\ESA_LUC_2000_SPEI03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        # print(f.variables['lccs_class'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        lcc = (f.variables['lcc'][:])
        
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


def exact_data3():
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
def read_nc2():
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

#%%
def read_nc3():
    inpath = r"E:/CO2/DENG/CO2_81_13_RG_SPEI0.5x0.5.nc"
    with nc.Dataset(inpath) as f:
        #print(f.variables.keys())
        
        lat3 = (f.variables['lat'][:])
        lon3 = (f.variables['lon'][:])
        co2 = (f.variables['co2'][:])
    
    ###########
    def region(data):
        t = np.arange(396)
        data_global = xr.DataArray(data, dims=['t', 'y', 'x'], coords=[
                                   t, lat3, lon3])  # 原SPEI-base数据
        data_rg = data_global.loc[:, 40:55, 100:125] 
        

        return data_rg
    
    ###########
    def mn_yr(data):
        q_mn = []
        for mn in range(12):
            q_ = []
            for yr in range(33):
                q_.append(data[mn])
                mn += 12
            q_mn.append(q_)
                
        q_mn = np.array(q_mn)
        return q_mn
    
    co2 = region(co2)
    co2_MG = mn_yr(co2)
    
    return co2_MG


#%% mask数组
def mask_lcc(x, data):
    lcc2 = np.array(lcc)
    
    np.place(lcc2, lcc2!=x, 1)
    np.place(lcc2, lcc2==x, 0)
    
    spei_ma = np.empty((30, 50))
    spei_ma = np.ma.array(spei_ma)
    
    a = ma.masked_array(data, mask=lcc2)
 
    return a

def mask_sig(data):
    fit_fp_mask = ma.masked_array(data, mask=fit_fp>0.05)
    fit_fp_g = mask_lcc(130, fit_fp_mask)
    
    return fit_fp_g
#%%
read_nc0()

gpp = exact_data3() #82-18
pre = read_nc2() #81-20
co2 = read_nc3() #81-13
del lat2, lon2
#########统一年份
gpp0 = gpp[1:33, 4, :, :]
gpp = gpp[1:33, 5, :, :]
pre = pre[4, 1:33, :, :]
co2 = co2[5, 1:, :, :]

#%% 拟合 六月测试
#### y:gpp, x1: pre, x2: co2, 
fit_r2 = np.zeros((30, 50))
fit_fp = np.zeros((30, 50))

itc, a, b, d = np.zeros((30, 50)), np.zeros((30, 50)), np.zeros((30, 50)), np.zeros((30, 50))

for r in range(30):
    for c in range(50):
        y = gpp[:, r, c]
        x1 = pre[:, r, c]
        x2 = co2[:, r, c]
        x3 = gpp0[:, r, c]
        df = pd.DataFrame(dict(y=y, x1=x1, x2=x2, x3=x3))
        if True in df.isnull().values:
            print("含有缺测值！")
        else:
            fit=sm.formula.ols('y~x1+x2+x3', data=df).fit() #####df格式/数组格式均可
            fit_r2[r, c] = fit.rsquared
            fit_fp[r, c] = fit.f_pvalue
            itc[r, c], a[r, c], b[r, c], d[r, c] = fit.params

#%% mask 看草地
fit_r2_mk = mask_sig(fit_r2)
itc_mk, a_mk, b_mk, d_mk = mask_sig(itc), mask_sig(a), mask_sig(b), mask_sig(d)