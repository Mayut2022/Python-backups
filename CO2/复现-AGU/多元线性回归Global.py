# %%
import FittingReadDataGlobal as frd
import CartopyPlot as cp
import statsmodels.api as SM  # 和中间的变量sm重合，故改成大写
import pandas as pd
import numpy as np
import cmaps
import warnings
warnings.filterwarnings("ignore")


# %%
'''
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

# %% 多元线性回归
fit_r2 = np.zeros((360,  720))
fit_fp = np.ones((360,  720))

fit_params = np.zeros((6, 360,  720))
fit_tp = np.ones((6, 360,  720))

y_fit = np.zeros((18, 360,  720))

var0.append("co2")
n = 0
for r in range(360):
    if r % 30 == 0:
        print(f"{r}x720 is done!")
    for c in range(720):
        y = sif_gs[:, r, c]
        for i, x in enumerate(var0):
            exec(f"x{i+1}={x}_gs[:, r, c]")
        df = pd.DataFrame(dict(y=y, x1=x1, x2=x2, x3=x3, x4=x4, x5=x5))
        if True in df.isnull().values:
            n += 1
            # print("含有缺测值！")
        else:
            fit = SM.formula.ols('y~x1+x2+x3+x4+x5',
                                 data=df).fit()  # df格式/数组格式均可
            y_fit[:, r, c] = fit.predict()                     
            fit_r2[r, c] = fit.rsquared
            fit_fp[r, c] = fit.f_pvalue
            fit_params[:, r, c] = fit.params  # 依次为截距、各变量的参数
            fit_tp[:, r, c] = fit.pvalues

del r, c, i, x

# %% 数据运行较慢，保存数据
# np.savez("./FitGlobal.npz", r2=fit_r2, fp=fit_fp, params=fit_params, tp=fit_tp)
'''


# %%
lcc = np.load("/mnt/e/MODIS/MCD12C1/MCD12C1_vegnonveg_0.5X0.5.npy")
_ = np.load("./FitGlobal.npz")
fit_r2 = _["r2"]
fit_fp = _["fp"]
fit_params = _["params"]
fit_tp = _["tp"]


# %%

def read_plot(data1, data2):
    data1[lcc == 0] = np.nan
    data2[lcc == 0] = np.nan
    print('5 percentile is: ', np.nanpercentile(data1, 5))
    print('95 percentile is: ', np.nanpercentile(data1, 95), "\n")

    levels = np.arange(0, 0.81, 0.05)
    cmap = cmaps.MPL_OrRd[:81]
    title = "Fit Rsqured"
    cp.plot2(data1, data2, levels, cmap, title)


# read_plot(fit_r2, fit_fp)

# %%
def read_plot2(data1, data2, var):
    print(var)

    data1[lcc == 0] = np.nan
    data2[lcc == 0] = np.nan

    print('5 percentile is: ', np.nanpercentile(data1, 5))
    print('95 percentile is: ', np.nanpercentile(data1, 95), "\n")

    if var == "pre":
        levels = np.arange(-0.001, 0.0011, 0.0001)
    elif var == "tmp":
        levels = np.arange(-0.03, 0.031, 0.002)
    elif var == "vpd":
        levels = np.arange(-0.03, 0.031, 0.002)
    elif var == "sr":
        levels = np.arange(-0.001, 0.0011, 0.0001)
    elif var == "co2":
        levels = np.arange(-0.001, 0.0011, 0.0001)

    cmap = "bwr"
    title = f"Fit Params {var}"
    cp.plot2(data1, data2, levels, cmap, title)


# %%
var0 = ["pre", "tmp", "vpd", "sr", "co2"]
for var, p, pt in zip(var0[:], fit_params[1:], fit_tp[1:]):
    read_plot2(p, pt, var)
# %%
