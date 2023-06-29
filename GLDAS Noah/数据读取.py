# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 09:10:26 2022
有v2.0 v2.1两套资料
lat是从小到大，但是数据lat是从大到小

@author: MaYutong
"""

import netCDF4 as nc
import numpy as np

def read_nc1():
    global sm0, sm1, sm2, sm3, lat, lon, time, t, sm
    inpath = (r"E:/GLDAS Noah/DATA_GLOBAL/SM_V2.0_81_00/GLDAS_NOAH025_M.A198101.020.nc4.SUB.nc4")
    with nc.Dataset(inpath) as f:
        print(f.variables.keys())
        print(f.variables['SoilMoi0_10cm_inst']) # units: 1 kg m-2 = 1 mm
        print("")
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        # lcc = (f.variables['lcc'][:])
        time = (f.variables['time'][:])
        t = nc.num2date(time, 'days since 1948-01-01 00:00:0.0').data
        sm0 = (f.variables['SoilMoi0_10cm_inst'][:])
        sm1 = (f.variables['SoilMoi10_40cm_inst'][:])
        sm2 = (f.variables['SoilMoi40_100cm_inst'][:])
        sm3 = (f.variables['SoilMoi100_200cm_inst'][:])
        
        sm = np.vstack((sm0, sm1, sm2, sm3))
        
def read_nc2():
    global lcc, lat2, lon2
    inpath = (r"E:/GLDAS Noah/SM_V2.1_00_20/GLDAS_NOAH025_M.A200001.021.nc4.SUB.nc4")
    with nc.Dataset(inpath) as f:
        print(f.variables.keys())
        print(f.variables['time'])
        lat2 = (f.variables['lat'][:])
        lon2 = (f.variables['lon'][:])
        # lcc = (f.variables['lcc'][:])

read_nc1()
# read_nc2()

#%%
import matplotlib.pyplot as plt
plt.figure(1, dpi=500)
plt.imshow(a, cmap='Blues', vmin=0, vmax=40, origin="lower")
plt.colorbar(shrink=0.75)
plt.show()















