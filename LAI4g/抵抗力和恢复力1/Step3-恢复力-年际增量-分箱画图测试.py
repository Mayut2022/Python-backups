# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 21:09:30 2023

@author: MaYutong
"""


import warnings
from matplotlib import font_manager
import matplotlib.dates as mdate
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import seaborn as sns
import xarray as xr

plt.rcParams['font.sans-serif'] = ['times new roman']  # 指定默认字体
warnings.filterwarnings("ignore")
# %%

df1 = pd.read_excel(r"E:/LAI4g/python_MG/抵抗力和恢复力/LAI Sig Change2.xlsx", sheet_name="Sig Up")
df2 = pd.read_excel(r"E:/LAI4g/python_MG/抵抗力和恢复力/LAI Sig Change2.xlsx", sheet_name="Sig Down")


# %%

df1_1 = df1[df1['Year'] < 1999]
df1_2 = df1[df1['Year'] >= 1999]

df2_1 = df2[df2['Year'] < 1999]
df2_2 = df2[df2['Year'] >= 1999]

#%%
def cut(data, bin):
    non_corr = data["LAI Diff"]*data["SPEI Diff"]
    non_count = non_corr[non_corr<0]
    data["non_corr"] = non_corr
    data["non_corr"][data["non_corr"]<=0]=np.nan
    data = data.dropna()
    
    b = data.sort_values(by=[f"SPEI Diff"])
    for i in range(len(bin)-1):
        c = b[np.logical_and(b[f"SPEI Diff"]>bin[i], b[f"SPEI Diff"]<bin[i+1])]
        d = c.quantile(q=[0.95, 0.9, 0.75, 0.5, 0.25, 0.1, 0.05])
        d = d[["LAI Diff", "SPEI Diff"]]
        exec(f'd.columns = ["LAI Diff{i}", "SPEI Diff{i}"]')
        if i==0:
            d_all = d
        else:
            d_all = pd.concat([d_all, d], axis=1)
            
    return d_all

#%%
bin1 = np.arange(0, 2.51, 0.25)
df1_2_cut = cut(df1_2, bin1)  ## 显著上升
bin2 = np.arange(-2.5, 0.1, 0.25)
df2_2_cut = cut(df2_2*-1, bin1)  ## 显著下降

#%%
def Xlabel():
    age = np.random.randint((10), size=(10))
    bin = np.arange(0, 2.51, 0.25)
    cats = pd.cut(age, bin, right=False)
    
    x = cats.categories
    return x

xlabel = Xlabel()

#%%
def plot_test():
    df2_plot = df1_2_cut.iloc[:, ::2]
    _ = df2_2_cut.iloc[:, ::2]
    df2_plot = df2_plot.append(_)
    df2_plot.index = [df2_plot.index, ["humid"]*7+["dry"]*7]

    df2_plot = df2_plot.stack()
    df2_plot = df2_plot.reset_index()
    df2_plot.columns = ["Percentile", "State", "SPEI Change", "LAI Change"]
    
    fig = plt.figure(1, figsize=(12, 8), dpi=500)
    ax = fig.add_subplot(111)
    
    palette = ["#1681FC", "#FD7F1E"]
    # boxprops = dict(edgecolor=palette[0], facecolor='none', linewidth=1.5)
    # kwargs = dict(showmeans=True, boxprops=boxprops)
    # palette2 = ["r", "k"]
    boxprops = dict(edgecolor='k', linewidth=1.5)
    capprops = dict(color='k', linewidth=1.5)
    whiskerprops = dict(color='k', linewidth=1.5)
    medianprops = dict(color='yellow', linewidth=1.5)
    meanprops = dict(markerfacecolor='white', color='k')
    kwargs = dict(width=0.6, showmeans=True, boxprops=boxprops, capprops=capprops, 
                  whiskerprops=whiskerprops, medianprops=medianprops, meanprops=meanprops)
    f = sns.boxplot(ax=ax, data=df2_plot, x="SPEI Change", y="LAI Change", hue="State", palette=palette,
                **kwargs)
    ax.set_title("Sig Change 99-20")
    ax.set_xticklabels(xlabel)
    
    plt.show()
    
plot_test()
