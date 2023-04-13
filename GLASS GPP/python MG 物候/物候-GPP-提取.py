# %%

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

import savitzky_golay as sg

from Fitting import double_logistic as dl
from Fitting import twoderive as dv

# %%
lat_RG = np.linspace(40.25, 54.75, 30)
lon_RG = np.linspace(100.25, 124.75, 50)


def read_nc(yr):
    inpath = rf"/mnt/e/GLASS-GPP/8-day MG 0.5X0.5 (phenology) /GLASS_GPP_{yr}.nc"
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())

        gpp = (f.variables['GPP'][:]).data

    return gpp


def sg_dbl(data):
    doys = np.arange(1, 365, 8)
    tt = np.arange(1, 366, 1)

    SOP1 = np.zeros((30, 50))
    EOP1 = np.zeros((30, 50))
    GSL1 = np.zeros((30, 50))

    POP = np.zeros((30, 50))

    SOP2 = np.zeros((30, 50))
    EOP2 = np.zeros((30, 50))
    GSL2 = np.zeros((30, 50))

    for r in range(30):
        if r % 30 == 0:
            print('still alive!')
        for c in range(50):
            if data[:, r, c].mean() < 0:
                SOP1[r, c] = -9999
                EOP1[r, c] = -9999
                GSL1[r, c] = -9999

                POP[r, c] = -9999

                SOP2[r, c] = -9999
                EOP2[r, c] = -9999
                GSL2[r, c] = -9999
            else:
                data_sg = sg.savitzky_golay(data[:, r, c], 15, 4)
                data_dbl = dl.fit_phenology_model_double_logistic(
                    data_sg, doys)

                SOP1[r, c], EOP1[r, c] = dv.derive(data_dbl, tt)
                GSL1[r, c] = EOP1[r, c]-SOP1[r, c]

                POP[r, c] = data_dbl.argmax()

                SOP2[r, c], EOP2[r, c] = dv.derive2(data_dbl, tt)
                GSL2[r, c] = EOP2[r, c]-SOP2[r, c]

    return SOP1, EOP1, GSL1, SOP2, EOP2, GSL2, POP


# %%


def CreatNC(data1, data2, data3, data4, data5, data6, data7, yr):

    new_NC = nc.Dataset(
        rf"/mnt/e/GLASS-GPP/8-day MG 0.5X0.5 (phenology) /dbl_derive_{yr}.nc", 'w', format='NETCDF4')

    new_NC.createDimension('lat', 30)
    new_NC.createDimension('lon', 50)

    var = new_NC.createVariable('SOP1', 'f', ("lat", "lon"))
    var = new_NC.createVariable('EOP1', 'f', ("lat", "lon"))
    var = new_NC.createVariable('GSL1', 'f', ("lat", "lon"))
    var = new_NC.createVariable('SOP2', 'f', ("lat", "lon"))
    var = new_NC.createVariable('EOP2', 'f', ("lat", "lon"))
    var = new_NC.createVariable('GSL2', 'f', ("lat", "lon"))
    var = new_NC.createVariable('POP', 'f', ("lat", "lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))

    new_NC.variables['SOP1'][:] = data1
    new_NC.variables['EOP1'][:] = data2
    new_NC.variables['GSL1'][:] = data3
    new_NC.variables['SOP2'][:] = data4
    new_NC.variables['EOP2'][:] = data5
    new_NC.variables['GSL2'][:] = data6
    new_NC.variables['POP'][:] = data7
    new_NC.variables['lat'][:] = lat_RG
    new_NC.variables['lon'][:] = lon_RG

    # 最后记得关闭文件
    new_NC.close()


# %%
if __name__ == "__main__":
    for yr in range(1982, 2019):
        gpp = read_nc(yr)
        sop1, eop1, gsl1, sop2, eop2, gsl2, pop = sg_dbl(gpp)
        CreatNC(sop1, eop1, gsl1, sop2, eop2, gsl2, pop, yr)
        print("%d phenology is done" % yr)
# %%
