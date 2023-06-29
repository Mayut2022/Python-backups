
# %%
import cmaps

import netCDF4 as nc
import numpy as np
import pandas as pd

import CartopyPlot as cp
# %%


def read_nc():
    inpath = ('E:/CO2/python-AGU-MG/FittingMG_SensitivityC5.nc')
    with nc.Dataset(inpath, mode='r') as f:

        # print(f.variables.keys(), "\n")
        # print(f.variables['fit_all'], "\n")

        spei = (f.variables['spei_sen'][:]).data  # units: 无
        tmp = (f.variables['tmp_sen'][:]).data  # units: Celsius degree
        co2 = (f.variables['co2_sen'][:]).data  # units: ppm()
        co2 = co2*10
        sos = (f.variables['sos_sen'][:]).data

        return spei, tmp, co2, sos


# %%
def read_plot(data1, data2, var, unit):
    
    print(var)

    print('5 percentile is: ', np.nanpercentile(data1, 5))
    print('95 percentile is: ', np.nanpercentile(data1, 95), "\n")

    if var == "spei":
        levels = np.arange(-8, 8.1, 0.5)
    elif var == "tmp":
        levels = np.arange(-5, 5.1, 0.5)
    elif var == "co2":
        levels = np.arange(-2., 2.1, 0.1)
    elif var == "sos":
        levels = np.arange(-1., 1.1, 0.1)

    var = var.upper()
    cmap = cmaps.MPL_RdYlGn
    title = f"GPP(GSL Sum) Sensitivity to {var} (gC m-2 per {unit})"
    cp.plot(data1, data2, levels, cmap, title)



# %%
if __name__ == "__main__":
    spei, tmp, co2, sos = read_nc()
    var = ["spei", "tmp", "co2", "sos"]
    unit = ["SPEI", "Cesius Degree", "10 ppm", "Day"]
    for x, u in zip(var, unit):
        xx = eval(f"{x}")
        read_plot(xx, x, x, u)

# %%
