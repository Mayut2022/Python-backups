# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 16:52:35 2022

@author: MaYutong
"""

import netCDF4 as nc
import cftime
import xarray as xr
import numpy as np
from sklearn import preprocessing

import warnings
warnings.filterwarnings("ignore")

# %%
def read_nc():
    global lat, lon
    inpath = rf"E:/GLDAS Noah/DATA_RG/SM_81_20_ORINGINAL_SPEI0.5x0.5.nc"
    with nc.Dataset(inpath, mode='r') as f:
        # print(f.variables.keys())
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        sm = (f.variables['sm'][:])
        
        sm = sm.reshape(40, 12, 4, 50, 100)
        
    return sm

# %%


def CreatNC(data):
    new_NC = nc.Dataset(
        rf"E:/GLDAS Noah/DATA_RG/SM_Zscore_81_20_SPEI0.5x0.5.nc",
        'w', format='NETCDF4')

    new_NC.createDimension('time', 480)
    new_NC.createDimension('layer', 4)
    new_NC.createDimension('lat', len(lat))
    new_NC.createDimension('lon', len(lon))

    var = new_NC.createVariable('sm', 'f', ("time", "layer", "lat", "lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))

    new_NC.variables['sm'][:] = data
    new_NC.variables['lat'][:] = lat
    new_NC.variables['lon'][:] = lon

    var.description = "1981.1-2020.12 SM 仅合并未处理原始数据 monthly mean"
    var.time = 'time = pd.date_range("1981-01-01", periods=480, freq="MS")'
    var.unit = "原单位为kg m-2, 1 kg m-2 = 1 mm month mean"
    var.missingvalue = "nan"

    new_NC.close()


#%%

def Zscore(data):
    tmp_mn_z = np.zeros((40, 12, 4, 50, 100))
    for mn in range(1, 13):
        print(f"Month{mn} is in processing!")
        for l in range(4):
            for r in range(50):
                for c in range(100):
                    a = data[:, mn-1, l, r, c]
                    tmp_mn_z[:, mn-1, l, r, c] = preprocessing.scale(a) #########
    return tmp_mn_z

# %%
sm = read_nc()
sm_z = Zscore(sm)
sm_z_re = sm_z.reshape(480, 4, 50, 100)
CreatNC(sm_z_re)
