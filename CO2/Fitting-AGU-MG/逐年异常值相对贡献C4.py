# %%
import netCDF4 as nc
import numpy as np
import xarray as xr

import ReadDataGlobal as rd

# %%
# gpp, sif, pre, pet, tmp, vpd, sr, co2, spei
gpp, _, _, _, tmp, _, _, co2, spei \
    = rd.data_exact(1982, 2018, 1, 12)


def read_nc(yr, var):
    inpath = rf"E:/GLASS-GPP/8-day MG 0.5X0.5 (phenology) /dbl_derive_{yr}.nc"
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        var = var.upper()
        data = (f.variables[f'{var}'][:]).data
        data[data == -9999] = np.nan
    return data


def data(var):
    var_all = []
    for yr in range(1982, 2019):
        var1 = read_nc(yr, var)
        var_all.append(var1)

    var_all = np.array(var_all)

    var_m = np.nanmean(var_all, axis=0)

    return var_m, var_all


_, sos = data("sop2")
# %% 提取每个格点生长季
_ = np.load("E:/CO2/复现-AGU/TemAbove0.npz")
sm, em = _["sm"], _["em"]

var1 = ["gpp"]
var0 = ["spei", "tmp", "co2"]

# %%


def sif_xarray(band1):
    mn = np.arange(37)
    lat = np.linspace(-89.75, 89.75, 360)
    lon = np.linspace(-179.75, 179.75, 720)
    sif = xr.DataArray(band1, dims=['mn', 'y', 'x'], coords=[mn, lat, lon])

    sif_MG = sif.loc[:, 40:55, 100:125]
    return np.array(sif_MG)

def sif_xarray2(band1):
    lat = np.linspace(-89.75, 89.75, 360)
    lon = np.linspace(-179.75, 179.75, 720)
    sif = xr.DataArray(band1, dims=['y', 'x'], coords=[lat, lon])

    sif_MG = sif.loc[40:55, 100:125]
    return np.array(sif_MG)

#%%
def gs1(ind, data):
    data_gs = np.empty((37, 360, 720))
    for r in range(360):
        for c in range(720):
            sm1, em1 = int(sm[r, c]), int(em[r, c])+1
            data_gs[:, r, c] = np.nanmean(data[:, sm1:em1, r, c], axis=ind)

    data_MG = sif_xarray(data_gs)

    return data_MG


def gs0(ind, data):
    data_gs = np.empty((37, 360, 720))
    for r in range(360):
        for c in range(720):
            sm1, em1 = int(sm[r, c]), int(em[r, c])+1
            data_gs[:, r, c] = np.nanmean(data[sm1:em1, :, r, c], axis=ind)

    data_MG = sif_xarray(data_gs)

    return data_MG


for x in var1:
    exec(f"{x} = gs1(1, {x})")

for x in var0:
    print(x)
    exec(f"{x} = gs0(0, {x})")

del sm, em, x


# %%


def read_nc():
    inpath = (r'FittingGlobalSensitivityC4.nc')
    with nc.Dataset(inpath, mode='r') as f:

        # print(f.variables.keys(), "\n")
        # print(f.variables['fit_all'], "\n")

        spei = (f.variables['spei_sen'][:]).data  # units: 无
        tmp = (f.variables['tmp_sen'][:]).data  # units: Celsius degree
        co2 = (f.variables['co2_sen'][:]).data  # units: ppm()
        # co2 = co2*10
        sos = (f.variables['sos_sen'][:]).data

        return spei, tmp, co2, sos


sen_spei, sen_tmp, sen_co2, sen_sos = read_nc()


# %% 计算逐年相对贡献率
gpp_anom = gpp-gpp[0, :]
var_all = var0+["sos"]+var1
for x in var_all[:4]:
    exec(f"{x}_diff = {x}-{x}[0, :]")
    exec(f"{x}_anom = {x}_diff*sen_{x}")
    exec(f"{x}_con = {x}_anom/gpp_anom")


# %% 保存数据


def CreatNC():
    new_NC = nc.Dataset(
        r"E:/CO2/Fitting-AGU-MG/FittingMG_Anom_Con2.nc", 'w', format='NETCDF4')

    yr = np.arange(37)
    new_NC.createDimension('yr', 37)
    new_NC.createDimension('lat', 30)
    new_NC.createDimension('lon', 50)

    for x in var_all[:4]:
        exec(f'var1 = new_NC.createVariable("{x}_anom", "f", ("yr", "lat", "lon"))')
        exec(f'new_NC.variables["{x}_anom"][:] = {x}_anom')

        exec(f'new_NC.createVariable("{x}_con", "f", ("yr", "lat", "lon"))')
        exec(f'new_NC.variables["{x}_con"][:] = {x}_con')
    
    var = locals()['var1']
        
    var.description = "fit_all原数据拟合，anom为逐年不同变量对gpp的影响值，con为贡献率大小"
    var.units = "Y(GPP)=X1(spei)+X2(tmp)+X3(co2)+X4(sos)"
    var.time = "1982.1-2018.12"

    # 最后记得关闭文件
    new_NC.close()


# CreatNC()
