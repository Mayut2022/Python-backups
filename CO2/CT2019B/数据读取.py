# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 10:44:37 2022

@author: MaYutong
"""
# %%
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import xarray as xr


def read_nc1(inpath):
    global lat, lon, t, lev, gph, p, tmp, u
    with nc.Dataset(inpath, mode='r') as f:

        print(f.variables.keys(), "\n")
        print(f.variables['co2'], "\n")
        print(f.variables['temperature'], "\n")
        print(f.variables['orography'], "\n")
        print(f.variables['u'], "\n")
        # print(f.variables['boundary'], "\n")
        # print(f.variables['blh'], "\n")
        # print(f.variables['time'], "\n")
        # print(f.variables['level'], "\n")

        # print(f.variables['time_components'], "\n")
        # time2 = (f.variables['decimal_time'][:])
        time = (f.variables['time'][:])
        t = nc.num2date(time, 'days since 2000-01-01 00:00:0.0').data

        lat = (f.variables['latitude'][:])
        lon = (f.variables['longitude'][:])
        co2 = (f.variables['co2'][:])  # units: mol m-2 s-1; monthly mean
        lev = (f.variables['level'][:])
        gph = (f.variables['co2'][:])
        p = (f.variables['pressure'][:])
        tmp = (f.variables['temperature'][:])
        u = (f.variables['u'][:])

        return co2


def read_nc2(inpath):
    global lat, lon, time, t
    with nc.Dataset(inpath, mode='r') as f:

        print(f.variables.keys(), "\n")
        print(f.variables['bg'], "\n")
        print(f.variables['bio'], "\n")
        print(f.variables['time'], "\n")
        print(f.variables['level'], "\n")

        time = (f.variables['time'][:])
        t = nc.num2date(time, 'days since 2000-01-01 00:00:0.0').data
        co2 = (f.variables['bio'][:])  # units: mol m-2 s-1; monthly mean

        return co2


# %%
inpath = r"/mnt/e/CO2/CT2019B/molefractions/CO2_total_monthly/CT2019B.molefrac_glb3x2_2005-10.nc"
inpath2 = r"/mnt/e/CO2/CT2019B/molefractions/CO2_total_monthly/CT2019B.molefrac_nam1x1_2000-01.nc"
inpath3 = r"/mnt/e/CO2/CT2019B/fluxes/CT2019B.flux1x1.2000-monthly.nc"
inpath4 = r"/mnt/e/CO2/CT2019B/fluxes/CT2019B.flux1x1.2000-mean.nc"
inpath5 = r"/mnt/e/CO2/CT2019B/molefractions/components/CT2019B.molefrac_components_glb3x2_2000-01-01.nc"

# bio = read_nc2(inpath3)
# bio = read_nc1(inpath4)

# co2 = read_nc1(inpath)
co2 = read_nc2(inpath5)


# %%
def region1(data):
    lev = np.arange(25)
    pre_rg1global = xr.DataArray(data, dims=['l', 'y', 'x'], coords=[
        lev, lat, lon])  # 原SPEI-base数据
    pre_rg1 = pre_rg1global.loc[:, 40:55, 100:125]

    return np.array(pre_rg1)


# a = co2[0, :, :, :]
co2 = np.squeeze(co2)
co2_MG = region1(co2)

# %%
a = co2[0, 0, :, :]
a[a == -1e+34] = np.nan

'''
b = gph[0, -1, :, :]

c = tmp[0, -1, :, :]

d = p[0, 10, :, :]
'''

plt.figure(1, dpi=500)
plt.imshow(a, cmap='jet', vmin=300, vmax=500)
plt.colorbar(shrink=0.75)
plt.show()

# %%
