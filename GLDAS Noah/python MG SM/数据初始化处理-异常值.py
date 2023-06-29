# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 16:52:35 2022

@author: MaYutong
"""

import netCDF4 as nc
import cftime
import xarray as xr
import numpy as np

#%%
def read_nc1():
    global lat, lon
    inpath = rf"E:/GLDAS Noah/DATA_RG/SM_81_20_ORINGINAL_SPEI0.5x0.5.nc"
    
    with nc.Dataset(inpath) as f:
        '''
        print(f.variables.keys())
        print(f.variables['SoilMoi0_10cm_inst']) # units: 1 kg m-2 = 1 mm
        print("")
        '''
        
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])

        sm = (f.variables['sm'][:])
        
        sm[sm==-9999] = np.nan
        return sm

# %%
def MG(data):
    t = np.arange(1, 41, 1)
    l = np.arange(1, 5, 1)
    spei_global = xr.DataArray(data, dims=['t', 'l', 'y', 'x'], coords=[
                               t, l, lat2, lon2])  # 原SPEI-base数据
    spei_rg1 = spei_global.loc[:, :, 40:55, 100:125]
    spei_rg1 = np.array(spei_rg1)
    return spei_rg1

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
    q_mn_ave = np.nanmean(q_mn, axis=1)
    
    return q_mn, q_mn_ave

#%%
def anom(data1, data2):
    q_anom = np.empty((480, 4, 50, 100))
    
    for l in range(4):
        for mn in range(12):
            yr = mn
            for i in range(40):
                yr = mn+12*i
                #print(yr, mn)
                q_anom[yr, l, :, :] = data1[yr, l, :, :]-data2[mn, l, :, :]
        
    return q_anom
    
#%%
sm = read_nc1()
#%% SM GLDAS
_, sm_mn_ave = mn_yr(sm)
sm_mn_anom = anom(sm, sm_mn_ave)

#%%
def CreatNC(data):
    new_NC = nc.Dataset(
        rf"E:/GLDAS Noah/DATA_RG/SM_Anom_81_20_ORINGINAL_SPEI0.5x0.5.nc", 
        'w', format='NETCDF4')
    
    new_NC.createDimension('time', 480)
    new_NC.createDimension('layer', 4)
    new_NC.createDimension('lat', len(lat))
    new_NC.createDimension('lon', len(lon))
    
    var = new_NC.createVariable('sm', 'f', ("time", "layer", "lat", "lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))
    
    new_NC.variables['sm'][:]=data
    new_NC.variables['lat'][:]=lat
    new_NC.variables['lon'][:]=lon
     
    var.description = "1981.1-2020.12 SM 仅合并未处理原始数据 monthly mean"
    var.time = 'time = pd.date_range("1981-01-01", periods=480, freq="MS")'
    var.unit = "kg m-2, 1 kg m-2 = 1 mm month mean"
    var.missingvalue = "nan"
    
    new_NC.close()

#CreatNC(sm_mn_anom)

#%%
import matplotlib.pyplot as plt

ind = 6
for yr in range(40):
    
    data = sm_mn_anom[ind, 1, :, :]
    print(np.nanmax(data), np.nanmin(data))
    ind += 12
    
    plt.figure(3, dpi=500)
    plt.imshow(data, cmap='RdBu', vmin=-20, vmax=20, origin="lower")
    plt.title(f"Year{yr+1981} July")
    plt.colorbar(shrink=0.75)
    plt.show()