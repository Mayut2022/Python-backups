# combin1 tmp, co2, spei, sos

# %%


import netCDF4 as nc
import numpy as np
import pandas as pd
import statsmodels.api as SM
import xarray as xr

import ReadDataGlobal as rd

import warnings
warnings.filterwarnings("ignore")

# %%
# gpp, sif, pre, pet, tmp, vpd, sr, co2, spei
gpp, _, _, _, tmp, _, _, co2, spei \
    = rd.data_exact(1982, 2018, 1, 12)

def read_nc(yr, var):
    inpath = rf"/mnt/e/GLASS-GPP/8-day MG 0.5X0.5 (phenology) /dbl_derive_{yr}.nc"
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
_ = np.load("/mnt/e/CO2/复现-AGU/TemAbove0.npz")
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
    exec(f"{x}_gs = gs1(1, {x})")

for x in var0:
    print(x)
    exec(f"{x}_gs = gs0(0, {x})")

del sm, em

# %% SPEI\T\CO2 按照第一年得到数据

def data00(data):
    data_gs = np.empty((37, 30, 50))
    for yr in range(18):
        data_gs[yr, :, :] = data[0, :, :]

    return data_gs


for x in var0:
    exec(f"{x}_00 = data00({x}_gs)")
    sos_00 = data00(sos)

# %%


def fit_values(Y, X1, X2, X3, X4):
    global n

    var = [X1, X2, X3, X4]
    y_fit = np.zeros((37, 30, 50))
    n = 0
    for r in range(30):
        if r % 30 == 0:
            print(f"{r}x50 is done!")
        for c in range(50):
            y = Y[:, r, c]
            for i, x in enumerate(var):
                exec(f"xx{i+1}=x[:, r, c]")

            x1, x2 = locals()['xx1'], locals()['xx2']
            # x3 = locals()['xx3']
            x3, x4 = locals()['xx3'], locals()['xx4']
            # x5 = locals()['xx5']

            df = pd.DataFrame(dict(y=y, x1=x1, x2=x2, x3=x3, x4=x4))  # , x4=x4, x5=x5
            if True in df.isnull().values:
                n += 1
                # print("含有缺测值！")
            else:
                fit = SM.formula.ols('y~x1+x2+x3+x4',
                                     data=df).fit()  # df格式/数组格式均可

                y_fit[:, r, c] = fit.predict()

    return y_fit


# %%
fit_all = fit_values(gpp_gs, spei_gs, tmp_gs, co2_gs, sos)

# %%
fit_spei00 = fit_values(gpp_gs, spei_00, tmp_gs, co2_gs, sos)
fit_tmp00 = fit_values(gpp_gs, spei_gs, tmp_00, co2_gs, sos)
fit_co200 = fit_values(gpp_gs, spei_gs, tmp_gs, co2_00, sos)
fit_sos00 = fit_values(gpp_gs, spei_gs, tmp_gs, co2_00, sos_00)

# %% 得到模拟差值
sos_gs = sos.copy()
var0.append("sos")
for x in var0:
    exec(f"{x}_df = {x}_gs-{x}_00")
    exec(f"fit_{x}_df = fit_all-fit_{x}00")

# %% Sensitivity


def fit_sen(Y, X):

    fit_params = np.zeros((2, 30,  50))

    for r in range(30):
        if r % 30 == 0:
            print(f"{r}x50 is done!")
        for c in range(50):
            y = Y[:, r, c]
            x = X[:, r, c]
            df = pd.DataFrame(dict(y=y, x=x))
            if True in df.isnull().values:
                pass
                # print("含有缺测值！")
            else:
                fit = SM.formula.ols('y~x',
                                     data=df).fit()  # df格式/数组格式均可
                fit_params[:, r, c] = fit.params  # 依次为截距、各变量的参数

    return fit_params


# %%
for x in var0:
    exec(f"{x}_sen = fit_sen(fit_{x}_df, {x}_df)")

# %% 保存数据


def CreatNC():
    new_NC = nc.Dataset(
        r"./FittingGlobalSensitivityC4.nc", 'w', format='NETCDF4')

    yr = np.arange(37)
    new_NC.createDimension('yr', 37)
    new_NC.createDimension('lat', 30)
    new_NC.createDimension('lon', 50)

    var = new_NC.createVariable("fit_all", "f", ("yr", "lat", "lon"))
    for x in var0:
        exec(f'new_NC.createVariable("fit_{x}00", "f", ("yr", "lat", "lon"))')
        exec(f'new_NC.variables["fit_{x}00"][:] = fit_{x}00')

        exec(f'new_NC.createVariable("{x}_sen", "f", ("lat", "lon"))')
        exec(f'new_NC.variables["{x}_sen"][:] = {x}_sen[1]')

    var.description = "fit_all原数据拟合，\
            fit_()00为X变量的敏感性试验得出来的拟合值，X_sen为其敏感性系数"
    var.units = "Y(sif)=X1(spei)+X2(tmp)+X3(co2)+X4(sos)"
    var.time = "1981.1-2018.12"

    # 最后记得关闭文件
    new_NC.close()


CreatNC()


# %%
