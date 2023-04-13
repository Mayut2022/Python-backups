
# %%

import cmaps
import numpy as np
import pandas as pd
import statsmodels.api as sm

import CartopyPlot as cp
import FittingReadData as frd


# %%
# 读取数据，00-18，growing season mean: 4-10
frd.data_time(2000, 2018, 4, 10)
ndvi, gpp, *x_data, co2_CT = frd.data_exact(2000, 2018, 4, 10)

# %%
ndvi_m, gpp_m = np.nanmean(ndvi, axis=1), np.nanmean(gpp, axis=1)
x_var = ["co2", "pre", "tmp", "vpd", "sr"]
for x, xd in zip(x_var, x_data):
    exec(f"{x}_m = np.nanmean(xd, axis=0)")
    del x, xd
co2_CT_m = np.nanmean(co2_CT, axis=1)


# %% 多元线性回归
fit_r2 = np.zeros((30, 50))
fit_fp = np.zeros((30, 50))

fit_params = np.zeros((6, 30, 50))
fit_tp = np.zeros((6, 30, 50))

x_var.append("co2_CT")
n = 0
for r in range(30):
    for c in range(50):
        y = gpp_m[:, r, c]
        for i, x in enumerate(x_var):
            exec(f"x{i+1}={x}_m[:, r, c]")
        df = pd.DataFrame(dict(y=y, x2=x2, x3=x3, x4=x4, x5=x5, x6=x6))
        if True in df.isnull().values:
            n += 1
            print("含有缺测值！")
        else:
            fit = sm.formula.ols('y~x2+x3+x4+x5+x6', data=df).fit()  # df格式/数组格式均可
            fit_r2[r, c] = fit.rsquared
            fit_fp[r, c] = fit.f_pvalue
            fit_params[:, r, c] = fit.params  # 依次为截距、各变量的参数
            fit_tp[:, r, c] = fit.pvalues

del r, c, i, x

# %%


def read_plot(data1, data2):
    print('5 percentile is: ', np.nanpercentile(data1, 5))
    print('95 percentile is: ', np.nanpercentile(data1, 95), "\n")

    levels = np.arange(0, 0.81, 0.05)
    cmap = cmaps.MPL_OrRd[:81]
    title = "Fit Rsqured"
    cp.plot(data1, data2, levels, cmap, title)


read_plot(fit_r2, fit_fp)


# %%
def read_plot(data1, data2, var):
    print('5 percentile is: ', np.nanpercentile(data1, 5))
    print('95 percentile is: ', np.nanpercentile(data1, 95), "\n")

    if var == "pre":
        levels = np.arange(-1, 1.1, 0.05)
    elif var == "tmp":
        levels = np.arange(-18, 18.1, 1)
    elif var == "vpd":
        levels = np.arange(-20, 20.1, 1)
    elif var == "sr":
        levels = np.arange(-0.7, 0.71, 0.05)
    elif var == "co2_CT":
        levels = np.arange(-0.5, 0.51, 0.05)
    cmap = "bwr"
    title = f"Fit Params {var}"
    cp.plot(data1, data2, levels, cmap, title)


# %%
for var, p, pt in zip(x_var[1:], fit_params[1:], fit_tp[1:]):
    read_plot(p, pt, var)
# %%
