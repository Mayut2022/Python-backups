
import os

import netCDF4 as nc
import numpy as np
import pandas as pd
import pprint
from pyhdf.SD import SD, SDC

import xarray as xr


def read_hdf(full_path):
    global data1
    hdf = SD(full_path)
    
    # print(hdf.info())
    # data = hdf.datasets()
    # for idx,sds in enumerate(data.keys()):
    # 	print (idx,sds)
    
    sds_obj = hdf.select('GPP')
    data1 = sds_obj.get()
    # pprint.pprint(sds_obj.attributes())

    return data1


# %%

lat = np.linspace(89.975, -89.975, 3600)
lon = np.linspace(-180, 179.95, 7200)

lat_spei = np.linspace(-89.75, 89.75, 360)
lon_spei = np.linspace(-179.75, 179.75, 720)

lat_RG = np.linspace(40.25, 54.75, 30)
lon_RG = np.linspace(100.25, 124.75, 50)


def avhrr_to_ESA(data):
    data_global = xr.DataArray(data, dims=['y', 'x'], coords=[
                               lat, lon])  # 原SPEI-base数据
    data_ESA = data_global.interp(y=lat_spei, x=lon_spei, method="linear")
    data_RG = data_ESA.loc[40:55, 100:125]

    return np.array(data_RG)

# %%


def os_data(t_str, yr):
    global gpp
    # 想要移动文件所在的根目录
    rootdir = rf"/mnt/e/GLASS-GPP/AVHRR/{yr}/"
    # 获取目录下文件名清单
    files = os.listdir(rootdir)

    for file in files:
        for i, x in enumerate(t_str):
            if str(yr)+x in file:  # 因为索要移动的文件名均有‘_’,因此利用此判断其是否是所需要移动的文件
                full_path = os.path.join(rootdir, file)  # 完整的路径
                print(full_path)
                gpp = read_hdf(full_path)

                gpp = avhrr_to_ESA(gpp)
                # print(gpp.shape)

                gpp = np.array(gpp, dtype="float64") ## 转换缺测值
                gpp[gpp == 65535] = np.nan ## 缺测值
                gpp = gpp*0.01 ## scale factor
                gpp = gpp.reshape(1, 30, 50)
                if i == 0:
                    gpp_all = gpp
                else:
                    gpp_all = np.vstack((gpp_all, gpp))

    return gpp_all


#%%
def CreatNC(yr, data):
    
    t = np.arange(1, 365, 8)
    
    new_NC = nc.Dataset(
        rf"/mnt/e/GLASS-GPP/8-day RG 0.5X0.5 (phenology) /GLASS_GPP_{yr}.nc", 
        'w', format='NETCDF4')
    
    new_NC.createDimension('t', 46)
    new_NC.createDimension('lat', 30)
    new_NC.createDimension('lon', 50)
    
    var=new_NC.createVariable('GPP', 'f', ("t","lat","lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))
    
    new_NC.variables['GPP'][:]=data
    new_NC.variables['lat'][:]=lat_RG
    new_NC.variables['lon'][:]=lon_RG
    
    var.description = \
        "Units: gC m-2 day-1; FillValue: 65535, 已处理为np.nan; \
        scale_factor: 0.01 (保存时已处理)"
    '''
    var.units = "gC m-2 day-1"
    var.FillValue = "65535, 已处理为np.nan"
    var.scale_factor = "0.01 (保存时已处理)"
    var.validrange = "[0, 3000]"
    '''
    #最后记得关闭文件
    new_NC.close()

# %%
if __name__ == "__main__":
    # inpath = r"/mnt/e/GLASS-GPP/AVHRR/1982/GLASS12B02.V40.A1982001.2019363.hdf"
    # data = read_hdf(inpath)
    t = np.arange(1, 365, 8)
    t_str = []
    [t_str.append(str(x).zfill(3)) for x in t]
    for yr in range(1982, 2019):
        gpp_yr = os_data(t_str, yr)
        CreatNC(yr, gpp_yr)


# %%
