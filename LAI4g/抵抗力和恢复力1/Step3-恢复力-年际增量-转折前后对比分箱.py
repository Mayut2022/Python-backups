# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 21:09:30 2023

转折前后同干旱和湿润对比

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

dfup = pd.read_excel(r"E:/LAI4g/python_MG/抵抗力和恢复力/LAI Sig Change2.xlsx", sheet_name="Sig Up")
dfdn = pd.read_excel(r"E:/LAI4g/python_MG/抵抗力和恢复力/LAI Sig Change2.xlsx", sheet_name="Sig Down")


# %%

dfup_1 = dfup[dfup['Year'] < 1999]
dfup_2 = dfup[dfup['Year'] >= 1999]

dfdn_1 = dfdn[dfdn['Year'] < 1999]
dfdn_2 = dfdn[dfdn['Year'] >= 1999]

#%%
def Xlabel():
    age = np.random.randint((10), size=(10))
    bin = np.arange(0, 2.76, 0.25)
    cats = pd.cut(age, bin, right=False)
    
    x = cats.categories
    return x

xlabel = Xlabel()

#%%
def cut(data, bin, cat):
    non_corr = data["LAI Diff"]*data["SPEI Diff"]
    non_count = non_corr[non_corr<0]
    data["non_corr"] = non_corr
    data["non_corr"][data["non_corr"]<=0]=np.nan
    data = data.dropna()
    
    b = data.sort_values(by=[f"SPEI Diff"])
    for i in range(len(bin)-1):
        c = b[np.logical_and(b[f"SPEI Diff"]>bin[i], b[f"SPEI Diff"]<bin[i+1])]
        c["Level"] = eval(f'["{str(i).zfill(2)}"]*len(c)')
        d = c.copy()
        d = d[["LAI Diff", "SPEI Diff", "Level"]]
        if i==0:
            d_all = d
        else:
            d_all = pd.concat([d_all, d], axis=0)
    d_all["State"] = [cat]*len(d_all)
    return d_all

#%%
def plot_box(data, title):
    
    fig = plt.figure(1, figsize=(12, 8), dpi=500)
    ax = fig.add_subplot(111)
    
    # palette = ["#1681FC", "#FD7F1E"]
    # palette = ["moccasin", "coral"]
    palette = ["lightskyblue", "dodgerblue"]
    boxprops = dict(edgecolor='k', linewidth=1.5)
    capprops = dict(color='k', linewidth=1.5)
    whiskerprops = dict(color='k', linewidth=1.5)
    medianprops = dict(color='k', linewidth=1.5)
    meanprops = dict(color='k', linewidth=1.5)
    kwargs = dict(width=0.6, showmeans=True, meanprops=meanprops, boxprops=boxprops, capprops=capprops, 
                  whiskerprops=whiskerprops, medianprops=medianprops, meanline=True)
    f = sns.boxplot(ax=ax, data=data, x="Level", y="LAI Diff", hue="State", palette=palette,
                **kwargs)
    
    ax.set_ylim(-0.05, 1.3)
    ax.set_xlabel("SPEI Change")
    ax.set_xticklabels(xlabel)
    ax.set_ylabel("LAI Change (Units: m2/m2)")
    
    ax.set_title(f"{title}")
    plt.savefig(rf'E:/LAI4g/python_MG/抵抗力和恢复力/JPG/{title}.jpg',
                bbox_inches='tight')
    
    plt.show()
    
    
def plot_bar(data, title):
    fig = plt.figure(1, figsize=(12, 8), dpi=500)
    ax = fig.add_subplot(111)
    
    # palette = ["#1681FC", "#FD7F1E"]
    # palette = ["moccasin", "coral"]
    palette = ["lightskyblue", "dodgerblue"]
    kwargs = dict(edgecolor='white', linewidth=1.5)
    sns.barplot(data=data, x="Level", y="LAI Diff", 
                hue="State", hue_order=["Before", "After"], palette=palette,
                **kwargs)
    
    ax.set_ylim(0, 260)
    ax.set_xlabel("SPEI Change")
    ax.set_xticklabels(xlabel)
    ax.set_ylabel("LAI Change Counts")
    
    ax.set_title(f"{title}")
    plt.savefig(rf'E:/LAI4g/python_MG/抵抗力和恢复力/JPG/{title}.jpg',
                bbox_inches='tight')
    plt.show()



#%%  干旱/显著下降对比
# bin1 = np.arange(0, 2.76, 0.25)
# dfdn_1_cut = cut(dfdn_1*-1, bin1, "Before")  ## 显著上升

# dfdn_2_cut = cut(dfdn_2*-1, bin1, "After")  ## 显著下降


# dfdn_plot = pd.concat([dfdn_1_cut, dfdn_2_cut])
# plot_box(dfdn_plot, title=r"Sig Down Comparison")

#%%
# dfdn_group = dfdn_plot["LAI Diff"].groupby([dfdn_plot['State'], dfdn_plot['Level']])
# dfdn_count = dfdn_group.count()
# dfdn_count = dfdn_count.reset_index()
# dfdn_count.sort_values(by=["State", "Level"])

# plot_bar(dfdn_count, title=r"Sig Down Comparison Sample Counts")



#%%  湿润/显著上升对比
bin1 = np.arange(0, 2.76, 0.25)
dfup_1_cut = cut(dfup_1, bin1, "Before")  ## 显著上升

dfup_2_cut = cut(dfup_2, bin1, "After")  ## 显著下降


dfup_plot = pd.concat([dfup_1_cut, dfup_2_cut])
plot_box(dfup_plot, title=r"Sig Up Comparison")

#%%
dfup_group = dfup_plot["LAI Diff"].groupby([dfup_plot['State'], dfup_plot['Level']])
dfup_count = dfup_group.count()
dfup_count = dfup_count.reset_index()
dfup_count.sort_values(by=["State", "Level"])

plot_bar(dfup_count, title=r"Sig Up Comparison Sample Counts")



