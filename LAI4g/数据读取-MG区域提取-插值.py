# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 10:43:55 2023

@author: MaYutong
"""
#%%
import netCDF4 as nc
import numpy as np
import os
import rasterio
from scipy import signal
from sklearn import preprocessing
import xarray as xr

# %%
_ = np.load("latlon.npz")
lat, lon = _["lat"], _["lon"]


def lai_xarray(band1):
    data = xr.DataArray(band1, dims=['y', 'x'], coords=[lat, lon])
    data_MG = data.loc[55:40, 100:125]
    return np.array(data_MG), data_MG.y, data_MG.x


def read_tif(inpath):
    global lat_MG, lon_MG
    with rasterio.open(inpath) as ds:
        band1 = ds.read(1)
    data, lat_MG, lon_MG = lai_xarray(band1)
    data = data*0.01
    data[data == 655.35] = np.nan

    return data


# %%


def lai_all(path):
    a = path.split('/')
    b = a[2].split('_')
    ys, ye = b[0], b[1]
    year = np.arange(int(ys), int(ye)+1)
    month = []
    [month.append(str(x).zfill(2)) for x in range(1, 13)]

    filename = os.listdir(path)

    lai_yr = []
    for yr in year[:]:
        lai_mn = []
        for mn in month:
            lai_ = []
            date = str(yr)+mn
            for file in filename:
                if date in file:
                    inpath = path+file
                    print(inpath)
                    lai = read_tif(inpath)
                    lai_.append(lai)  # 每个月两天
            lai_ = np.array(lai_)
            lai_ = np.nanmean(lai_, axis=0)
            lai_mn.append(lai_)  # 每年数据
        lai_yr.append(lai_mn)
    lai_yr = np.array(lai_yr)

    return lai_yr


path1 = r"/mnt/e/LAI4g/1982_1990_TIFF/"
path2 = r"/mnt/e/LAI4g/1991_2000_TIFF/"
path3 = r"/mnt/e/LAI4g/2001_2010_TIFF/"
path4 = r"/mnt/e/LAI4g/2011_2020_TIFF/"


for i in range(1, 5):
    data = eval(f"lai_all(path{i})")
    if i == 1:
        data_all = data
    else:
        data_all = np.vstack((data_all, data))

'''
数据有问题，1994年7月两数据都没有，因此循环会出问题
200605和200908两个月数据分别缺一个，则程序运行无问题
list append 函数：(180, 300)->(1, 180, 300)
'''

# %%生成新的nc文件


def CreatNC(data):
    year = np.arange(1982, 2021, 1)
    month = np.arange(1, 13, 1)

    new_NC = nc.Dataset(
        r"/mnt/e/LAI4g/data_MG/LAI_82_20_MG.nc", 'w', format='NETCDF4')

    new_NC.createDimension('year', 39)
    new_NC.createDimension('month', 12)
    new_NC.createDimension('lat', 180)
    new_NC.createDimension('lon', 300)

    var = new_NC.createVariable('lai', 'f', ("year", "month", "lat", "lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))

    new_NC.variables['lai'][:] = data
    new_NC.variables['lat'][:] = lat_MG
    new_NC.variables['lon'][:] = lon_MG

    var.data = "scale factor和缺测值均已处理，65535为np.nan"
    var.units = "m2/m2 (month mean)"

    # 最后记得关闭文件
    new_NC.close()

# CreatNC(data_all)

# %% 数据插值成SPEI精度


def GIMMS_to_SPEI(data, lat, lon):
    lat_spei = np.linspace(40.25, 54.75, 30)
    lon_spei = np.linspace(100.25, 124.75, 50)

    t = np.arange(1, 40, 1)
    mn = np.arange(1, 13, 1)
    ndvi = xr.DataArray(data, dims=['t', 'mn', 'y', 'x'], coords=[
        t, mn, lat, lon])  # 原SPEI-base数据

    ndvi_SPEI = ndvi.interp(t=t, mn=mn, y=lat_spei,
                            x=lon_spei, method="linear")
    lat_SPEI = ndvi_SPEI.y
    lon_SPEI = ndvi_SPEI.x

    return ndvi_SPEI, lat_SPEI, lon_SPEI


lai_SPEI, lat_SPEI, lon_SPEI = GIMMS_to_SPEI(data_all, lat_MG, lon_MG)

# %%


def CreatNC2(data):
    year = np.arange(1982, 2021, 1)
    month = np.arange(1, 13, 1)

    new_NC = nc.Dataset(
        r"/mnt/e/LAI4g/data_MG/LAI_Zscore_82_20_MG_SPEI0.5x0.5.nc", 'w', format='NETCDF4')

    new_NC.createDimension('year', 39)
    new_NC.createDimension('month', 12)
    new_NC.createDimension('lat', 30)
    new_NC.createDimension('lon', 50)

    var = new_NC.createVariable('lai', 'f', ("year", "month", "lat", "lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))

    new_NC.variables['lai'][:] = data
    new_NC.variables['lat'][:] = lat_SPEI
    new_NC.variables['lon'][:] = lon_SPEI

    var.data = "scale factor和缺测值均已处理，65535为np.nan"
    var.units = "m2/m2 (month mean)"
    var.description = "注意lat顺序，原顺序为由北向南，现顺序为由南向北"

    # 最后记得关闭文件
    new_NC.close()

# CreatNC2(lai_SPEI)


#%% Z-score归一化值

def Zscore(data):
    data_z = np.zeros((39, 12, 30, 50))
    for mn in range(12):
        for r in range(30):
            if r%10 == 0:
                print(f"columns {r} is done!")
            for c in range(50):
                data_z[:, mn, r, c] = preprocessing.scale(data[:, mn, r, c]) #########
    return data_z    
    
    
# lai_SPEI_z = Zscore(lai_SPEI)
# CreatNC2(lai_SPEI_z)


#%% 读取文件后，生成去趋势文件
def read_lai():
    global lat, lon
    inpath = (r"E:/LAI4g/data_MG/LAI_82_20_MG_SPEI0.5x0.5.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        # print(f.variables['lai'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        lai = (f.variables['lai'][:])
    return lai

lai = read_lai()

# %% 数据去趋势 1982-2020


def detrend(data):
    data_detrend = data.copy()
    for mn in range(12):
        for r in range(30):
            for c in range(50):
                test = data_detrend[:, mn, r, c]
                contain_nan = (True in np.isnan(test))
                if contain_nan == True:
                    continue
                else:
                    detrend = signal.detrend(test, type='linear')
                    data_detrend[:, mn, r, c] = detrend+test.mean()

    return data_detrend

lai_de = detrend(lai)
# lai_z = Zscore(lai)
# %%
def CreatNC3(data):
    year = np.arange(1982, 2021, 1)
    month = np.arange(1, 13, 1)

    # new_NC = nc.Dataset(
    #     r"/mnt/e/LAI4g/data_MG/LAI_Detrend_82_20_MG_SPEI0.5x0.5.nc", 'w', format='NETCDF4')
    new_NC = nc.Dataset(
        r"/mnt/e/LAI4g/data_MG/LAI_Detrend_Zscore_82_20_MG_SPEI0.5x0.5.nc", 'w', format='NETCDF4')

    new_NC.createDimension('year', 39)
    new_NC.createDimension('month', 12)
    new_NC.createDimension('lat', 30)
    new_NC.createDimension('lon', 50)

    var = new_NC.createVariable('lai', 'f', ("year", "month", "lat", "lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))

    new_NC.variables['lai'][:] = data
    new_NC.variables['lat'][:] = lat
    new_NC.variables['lon'][:] = lon

    var.data = "scale factor和缺测值均已处理，65535为np.nan"
    var.units = "m2/m2 (month mean)"
    var.description = "注意lat顺序，原顺序为由北向南，现顺序为由南向北"

    # 最后记得关闭文件
    new_NC.close()

# CreatNC3(lai_de)
# CreatNC3(lai_z)

# %%
def CreatNC4(data):
    year = np.arange(1982, 2021, 1)
    month = np.arange(1, 13, 1)

    new_NC = nc.Dataset(
        r"E:/LAI4g/data_MG/LAI_Anomaly_82_20_MG_SPEI0.5x0.5.nc", 'w', format='NETCDF4')

    new_NC.createDimension('year', 39)
    new_NC.createDimension('month', 12)
    new_NC.createDimension('lat', 30)
    new_NC.createDimension('lon', 50)

    var = new_NC.createVariable('lai', 'f', ("year", "month", "lat", "lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))

    new_NC.variables['lai'][:] = data
    new_NC.variables['lat'][:] = lat
    new_NC.variables['lon'][:] = lon

    var.data = "scale factor和缺测值均已处理，65535为np.nan"
    var.units = "m2/m2 (month mean)"
    var.description = "注意lat顺序，原顺序为由北向南，现顺序为由南向北"

    # 最后记得关闭文件
    new_NC.close()



# %%
def anomaly(data):
    data_anom = np.zeros((39, 12, 30, 50))
    data_ave = np.nanmean(data, axis=0)
    for yr in range(39):
        for mn in range(12):
            for r in range(30):
                for c in range(50):
                    data_anom[yr, mn, r, c] = data[yr, mn, r, c]-data_ave[mn, r, c]
                    
    return data_anom

# lai_anom = anomaly(lai_de)
lai_anom = anomaly(lai)
CreatNC4(lai_anom)
