# combin1 tmp, co2, spei
# combin2 pre, tmp, co2, spei
# combin3 pre, pet, tmp, co2
# combin4 (MG区域) tmp, co2, spei, sos
# combin5 (MG区域) tmp, co2, spei, sos 季节重新调整，gpp为5-9月生长季总和，其他为生长季平均

# %%

import cmaps
import netCDF4 as nc
import numpy as np
import pandas as pd
import statsmodels.api as SM
import xarray as xr

# import CartopyPlot as cp
import ReadDataGlobal as rd

import warnings
warnings.filterwarnings("ignore")


# %%


def read_lcc():
    inpath = (r"E:/ESA/ESA_LUC_2000_SPEI03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        # print(f.variables['lccs_class'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        lcc = (f.variables['lcc'][:])

    return lcc

lcc = read_lcc()
mask = lcc!=130
a_mk = np.ma.masked_array(a, mask=mask)
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
            data_gs[:, r, c] = np.nansum(data[:, 4:9, r, c], axis=ind)

    data_MG = sif_xarray(data_gs)

    return data_MG


def gs0(ind, data):
    data_gs = np.empty((37, 360, 720))
    for r in range(360):
        for c in range(720):
            data_gs[:, r, c] = np.nanmean(data[4:9, :, r, c], axis=ind)

    data_MG = sif_xarray(data_gs)

    return data_MG


for x in var1:
    print(x)
    exec(f"{x}_gs = gs1(1, {x})")

for x in var0:
    print(x)
    exec(f"{x}_gs = gs0(0, {x})")

# %%


def fit_values(Y, X1, X2, X3, X4):
    global n

    var = [X1, X2, X3, X4]
    fit_r2 = np.zeros((30, 50))
    fit_fp = np.ones((30, 50))

    fit_params = np.zeros((5, 30, 50))
    fit_tp = np.ones((5, 30, 50))
    n = 0
    for r in range(30):
        if r % 30 == 0:
            print(f"{r}x720 is done!")
        for c in range(50):
            y = Y[:, r, c]
            for i, x in enumerate(var):
                exec(f"xx{i+1}=x[:, r, c]")

            x1, x2 = locals()['xx1'], locals()['xx2']
            # x3 = locals()['xx3']
            x3, x4 = locals()['xx3'], locals()['xx4']
            # x5 = locals()['xx5']

            df = pd.DataFrame(dict(y=y, x1=x1, x2=x2, x3=x3, x4=x4))  # , x5=x5
            if True in df.isnull().values:
                n += 1
                # print("含有缺测值！")
            else:
                fit = SM.formula.ols('y~x1+x2+x3+x4',
                                     data=df).fit()  # df格式/数组格式均可
                fit_r2[r, c] = fit.rsquared
                fit_fp[r, c] = fit.f_pvalue
                fit_params[:, r, c] = fit.params  # 依次为截距、各变量的参数
                fit_tp[:, r, c] = fit.pvalues

    return fit_r2, fit_fp, fit_params, fit_tp


# %% 数据运行较慢，保存数据
fit_r2, fit_fp, fit_params, fit_tp = fit_values(
    gpp_gs, spei_gs, tmp_gs, co2_gs, sos)
np.savez("E:/CO2/python-AGU-MG/FitCombin5.npz", r2=fit_r2,
         fp=fit_fp, params=fit_params, tp=fit_tp)


# %%
lcc = np.load("E:/MODIS/MCD12C1/MCD12C1_vegnonveg_0.5X0.5.npy")
_ = np.load("./FitCombin4.npz")
fit_r2 = _["r2"]
fit_fp = _["fp"]
fit_params = _["params"]
fit_tp = _["tp"]

# %%


def read_plot(data1, data2):
    # data1[lcc == 0] = np.nan
    # data2[lcc == 0] = np.nan
    print('5 percentile is: ', np.nanpercentile(data1, 5))
    print('95 percentile is: ', np.nanpercentile(data1, 95), "\n")

    levels = np.arange(0, 0.81, 0.05)
    cmap = cmaps.MPL_YlOrRd
    title = "C4 Fit Rsqured (82-18)"
    # cp.plot(data1, data2, levels, cmap, title)


read_plot(fit_r2, fit_fp)

# %%


def read_plot2(data1, data2, var):
    print(var)

    # data1[lcc == 0] = np.nan
    # data2[lcc == 0] = np.nan

    print('5 percentile is: ', np.nanpercentile(data1, 5))
    print('95 percentile is: ', np.nanpercentile(data1, 95), "\n")

    if var == "spei":
        levels = np.arange(-10, 10.1, 0.5)
    elif var == "tmp":
        levels = np.arange(-10, 10.1, 0.5)
    # elif var == "pre":
    #     levels = np.arange(-1, 1.01, 0.05)
    # elif var == "pet":
    #     levels = np.arange(-80, 81, 5)
    elif var == "co2":
        levels = np.arange(-0.3, 0.31, 0.05)
    elif var == "sos":
        levels = np.arange(-1.5, 1.51, 0.25)

    cmap = "bwr"
    title = f"C4 Fit Params {var}"
    cp.plot(data1, data2, levels, cmap, title)


# %%
var0.append("sos")
for var, p, pt in zip(var0[:], fit_params[1:], fit_tp[1:]):
    read_plot2(p, pt, var)

# %%
