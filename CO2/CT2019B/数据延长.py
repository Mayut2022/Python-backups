# %%
import cmaps
import netCDF4 as nc
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xarray as xr

import CartopyPlot as cp
# %%
lat_spei = np.linspace(-89.75, 89.75, 360)
lon_spei = np.linspace(-179.75, 179.75, 720)


def read_nc():
    inpath = (
        '/mnt/e/CO2/CT2019B/molefractions/XCO2 Global/XCO2_00_18_SPEI0.5x0.5.nc')
    with nc.Dataset(inpath, mode='r') as f:

        print(f.variables.keys(), "\n")
        co2 = (f.variables['co2'][:14, :, ]).data  # units: ppm
        co2 = co2.reshape(14*12, 360, 720)

        co2_pro = (f.variables['co2'][14:, :, ]).data  # units: ppm
        co2_pro = co2_pro.reshape(5*12, 360, 720)

        return co2, co2_pro


def read_nc2():
    inpath = ('/mnt/e/CO2/DENG/CO2_81_13_Global_SPEI0.5x0.5.nc')
    with nc.Dataset(inpath, mode='r') as f:

        print(f.variables.keys(), "\n")
        co2 = (f.variables['co2'][228:]).data  # units: ppm
        co2_all = (f.variables['co2'][:]).data

        return co2, co2_all


# %%
t1 = pd.date_range('2000/01', periods=228, freq="MS")  # 00-18: 0-168
t2 = pd.date_range('1981/01', periods=396, freq="MS")  # 00-18: 228-396
co2_ct, co2_ct_pro = read_nc()
co2_d, co2_d_all = read_nc2()


# %%
def fit(Y, X):
    fit_r2 = np.zeros((360,  720))
    fit_fp = np.ones((360,  720))

    fit_params = np.zeros((2, 360,  720))
    # fit_tp = np.ones((2, 360,  720))

    for r in range(360):
        if r % 30 == 0:
            print(f"{r}x720 is done!")
        for c in range(720):
            y = Y[:, r, c]
            x = X[:, r, c]
            df = pd.DataFrame(dict(y=y, x=x))
            if True in df.isnull().values:
                pass
                # print("含有缺测值！")
            else:
                fit = sm.formula.ols('y~x',
                                     data=df).fit()  # df格式/数组格式均可

                fit_r2[r, c] = fit.rsquared
                fit_fp[r, c] = fit.f_pvalue
                fit_params[:, r, c] = fit.params  # 依次为截距、各变量的参数
                # fit_tp[:, r, c] = fit.pvalues

    return fit_r2, fit_fp, fit_params


# %%
# fit_r2, fit_fp, fit_params = fit(co2_d, co2_ct)

# %%
# np.savez("./co2prolong", r2=fit_r2, fp=fit_fp, params=fit_params)
_ = np.load("./co2prolong.npz")
fit_r2, fit_fp, fit_params = _["r2"], _["fp"], _["params"]
# %%


def read_plot(data1, data2):
    data1[lcc == 0] = np.nan
    data2[lcc == 0] = np.nan
    print('5 percentile is: ', np.nanpercentile(data1, 5))
    print('95 percentile is: ', np.nanpercentile(data1, 95), "\n")

    levels = np.arange(0.8, 1.01, 0.02)
    cmap = cmaps.MPL_OrRd[:81]
    title = "Fit Rsqured Prolong (Y: Deng X: CT)"
    cp.plot2(data1, data2, levels, cmap, title)


def read_plot2(data1, data2):

    data1[lcc == 0] = np.nan
    data2[lcc == 0] = np.nan

    print('5 percentile is: ', np.nanpercentile(data1, 5))
    print('95 percentile is: ', np.nanpercentile(data1, 95), "\n")

    levels = np.arange(0.9, 1.10, 0.02)
    cmap = "bwr"
    title = f"Fit Params (Y: Deng X: CT)"
    cp.plot2(data1, data2, levels, cmap, title)


lcc = np.load("/mnt/e/MODIS/MCD12C1/MCD12C1_vegnonveg_0.5X0.5.npy")
# read_plot(fit_r2, fit_fp)
# read_plot2(fit_params[1], fit_fp)

# %%
co2_pro = fit_params[0]+fit_params[1]*co2_ct_pro
# %%


def corr(Y, X):
    global n
    fit_r2 = np.zeros((360,  720))

    n = 1
    for r in range(360):
        if r % 30 == 0:
            print(f"{r}x720 is done!")
        for c in range(720):
            y = Y[:, r, c]
            x = X[:, r, c]
            df = pd.DataFrame(dict(y=y, x=x))
            if True in df.isnull().values:
                n += 1
                # print("含有缺测值！")
            else:
                rr = df.corr(method="pearson")
                fit_r2[r, c] = rr.iloc[0, 1]

    return fit_r2*fit_r2


# %%
# pro_r2 = corr(co2_pro, co2_ct_pro)
# %%
# read_plot(pro_r2, fit_fp)
# %%
def CreatNC3(data):
    new_NC = nc.Dataset(
        r"/mnt/e/CO2/DENG/CO2_81_18_Global_Prolong_SPEI0.5x0.5.nc", 'w', format='NETCDF4')

    new_NC.createDimension('time', 456)
    new_NC.createDimension('lat', 360)
    new_NC.createDimension('lon', 720)

    var = new_NC.createVariable('co2', 'f', ("time", "lat", "lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))

    new_NC.variables['co2'][:] = data
    new_NC.variables['lat'][:] = lat_spei
    new_NC.variables['lon'][:] = lon_spei

    var.data = "scale factor和缺测值均未处理，似乎不用处理"
    var.units = "units: ppm; 1ppm=10^-6mol/mol=1μmol/mol; the average of the month"
    var.time = "81-18, 81-13为Deng的数据, 14-18为CT回归延长的数据"

    # 最后记得关闭文件
    new_NC.close()


co2_8118 = np.vstack((co2_d_all, co2_pro))
# CreatNC3(co2_8118)
# %%


def region(data):
    t = np.arange(456)
    lat_rg = np.linspace(35.25, 59.75, 50)
    lon_rg = np.linspace(100.25, 149.75, 100)
    data_global = xr.DataArray(data, dims=['t', 'y', 'x'], coords=[
                               t, lat_spei, lon_spei])  # 原SPEI-base数据
    data_rg = data_global.loc[:, 35:60, 100:150]  # 不然边界没有值

    return data_rg, lat_rg, lon_rg


def CreatNC(data):
    new_NC = nc.Dataset(
        r"/mnt/e/CO2/DENG/CO2_81_18_RG_Prolong_SPEI0.5x0.5.nc", 'w', format='NETCDF4')

    new_NC.createDimension('time', 456)
    new_NC.createDimension('lat', 50)
    new_NC.createDimension('lon', 100)

    var = new_NC.createVariable('co2', 'f', ("time", "lat", "lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))

    new_NC.variables['co2'][:] = data
    new_NC.variables['lat'][:] = lat_rg
    new_NC.variables['lon'][:] = lon_rg

    var.ndvi = ("包含此前尝试的三个区域，region1 ESA IM")
    var.data = "scale factor和缺测值均未处理，似乎不用处理"
    var.units = "units: ppm; 1ppm=10^-6mol/mol=1μmol/mol; the average of the month"
    var.time = "81-18, 81-13为Deng的数据, 14-18为CT回归延长的数据"

    # 最后记得关闭文件
    new_NC.close()

# %%
co2_8118_rg, lat_rg, lon_rg = region(co2_8118)
CreatNC(co2_8118_rg)
# %%
