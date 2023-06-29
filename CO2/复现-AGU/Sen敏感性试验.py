
# VPD\T\CO2 对 SIF 的敏感性试验

# %%
import FittingReadDataGlobal as frd
import CartopyPlot as cp
import statsmodels.api as SM  # 和中间的变量sm重合，故改成大写
import pandas as pd
import netCDF4 as nc
import numpy as np
import cmaps

import warnings
warnings.filterwarnings("ignore")
# %%
frd.data_time(2001, 2018, 1, 12)
sif, pre, tmp, vpd, sr, co2 = frd.data_exact(2001, 2018, 1, 12)

# %% 提取每个格点生长季
_ = np.load("/mnt/e/CO2/复现-AGU/TemAbove0.npz")
sm, em = _["sm"], _["em"]

var1 = ["co2", "sif"]
var0 = ["pre", "tmp", "vpd", "sr"]

# %%


def gs1(ind, data):
    data_gs = np.empty((18, 360, 720))
    for r in range(360):
        for c in range(720):
            sm1, em1 = int(sm[r, c]), int(em[r, c])+1
            data_gs[:, r, c] = np.nanmean(data[:, sm1:em1, r, c], axis=ind)

    return data_gs


def gs0(ind, data):
    data_gs = np.empty((18, 360, 720))
    for r in range(360):
        for c in range(720):
            sm1, em1 = int(sm[r, c]), int(em[r, c])+1
            data_gs[:, r, c] = np.nanmean(data[sm1:em1, :, r, c], axis=ind)

    return data_gs


for x in var1:
    exec(f"{x}_gs = gs1(1, {x})")

for x in var0:
    print(x)
    exec(f"{x}_gs = gs0(0, {x})")

# %% VPD\T\CO2 按照第一年得到数据


def data00(data):
    data_gs = np.empty((18, 360, 720))
    for yr in range(18):
        data_gs[yr, :, :] = data[0, :, :]

    return data_gs


var00 = ["vpd", "tmp", "co2"]
for x in var00:
    exec(f"{x}_00 = data00({x}_gs)")

# %%


def fit_values(Y, X1, X2, X3, X4, X5):
    var = [X1, X2, X3, X4, X5]
    y_fit = np.zeros((18, 360, 720))
    n = 0
    for r in range(360):
        if r % 30 == 0:
            print(f"{r}x720 is done!")
        for c in range(720):
            y = Y[:, r, c]
            for i, x in enumerate(var):
                exec(f"xx{i+1}=x[:, r, c]")

            x1, x2 = locals()['xx1'], locals()['xx2']
            x3, x4 = locals()['xx3'], locals()['xx4']
            x5 = locals()['xx5']

            df = pd.DataFrame(dict(y=y, x1=x1, x2=x2, x3=x3, x4=x4, x5=x5))
            if True in df.isnull().values:
                n += 1
                # print("含有缺测值！")
            else:
                fit = SM.formula.ols('y~x1+x2+x3+x4+x5',
                                     data=df).fit()  # df格式/数组格式均可
                y_fit[:, r, c] = fit.predict()

    return y_fit


# %%
fit_all = fit_values(sif_gs, pre_gs, tmp_gs, vpd_gs, sr_gs, co2_gs)

# %%
fit_vpd00 = fit_values(sif_gs, pre_gs, tmp_gs, vpd_00, sr_gs, co2_gs)
fit_tmp00 = fit_values(sif_gs, pre_gs, tmp_00, vpd_gs, sr_gs, co2_gs)
fit_co200 = fit_values(sif_gs, pre_gs, tmp_gs, vpd_gs, sr_gs, co2_00)

# %% 得到模拟差值
for x in var00:
    exec(f"{x}_df = {x}_gs-{x}_00")
    exec(f"fit_{x}_df = fit_all-fit_{x}00")

# %% Sensitivity


def fit_sen(Y, X):
    # fit_r2 = np.zeros((360,  720))
    # fit_fp = np.ones((360,  720))

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
                fit = SM.formula.ols('y~x',
                                     data=df).fit()  # df格式/数组格式均可
                # y_fit = fit.predict()
                # fit_r2[r, c] = fit.rsquared
                # fit_fp[r, c] = fit.f_pvalue
                fit_params[:, r, c] = fit.params  # 依次为截距、各变量的参数
                # fit_tp[:, r, c] = fit.pvalues

    return fit_params


# %%
for x in var00:
    exec(f"{x}_sen = fit_sen(fit_{x}_df, {x}_df)")

# %% 保存数据


def CreatNC():
    new_NC = nc.Dataset(
        r"./FittingGlobalSensitivity.nc", 'w', format='NETCDF4')

    yr = np.arange(18)
    new_NC.createDimension('yr', 18)
    new_NC.createDimension('lat', 360)
    new_NC.createDimension('lon', 720)

    var = new_NC.createVariable("fit_all", "f", ("yr", "lat", "lon"))
    for x in var00:
        exec(f'new_NC.createVariable("fit_{x}00", "f", ("yr", "lat", "lon"))')
        exec(f'new_NC.variables["fit_{x}00"][:] = fit_{x}00')

        exec(f'new_NC.createVariable("{x}_sen", "f", ("lat", "lon"))')
        exec(f'new_NC.variables["{x}_sen"][:] = {x}_sen[1]')

    var.description = "fit_all原数据拟合，\
            fit_()00为X变量的敏感性试验得出来的拟合值，X_sen为其敏感性系数"
    var.units = "Y(sif)=pre+sr+X1(vpd)+X2(tmp)+X3(co2)"
    var.time = "2001.1-2018.12"

    # 最后记得关闭文件
    new_NC.close()

CreatNC()

# %% 简单画图
def read_plot(data1, data2, var):
    print(var)

    data1[lcc == 0] = np.nan
    data2[lcc == 0] = np.nan

    print('5 percentile is: ', np.nanpercentile(data1, 5))
    print('95 percentile is: ', np.nanpercentile(data1, 95), "\n")

    if var == "vpd":
        levels = [-0.001, 0, 0.001]
    elif var == "tmp":
        levels = [-0.01, 0, 0.01]
    elif var == "co2":
        levels = [-0.005, 0, 0.005]

    cmap = cmaps.cmocean_curl_r
    title = f"GOSIF sensitivity to {var}"
    # cp.plot2(data1, data2, levels, cmap, title)

lcc = np.load("/mnt/e/MODIS/MCD12C1/MCD12C1_vegnonveg_0.5X0.5.npy")
for x in var00:
    data1 = eval(f"{x}_sen[1]")
    data2 = lcc
    # read_plot(data1, data2, x)
# %%
