# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 21:09:30 2023

干旱和湿润对比

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
dfup["CR"] = dfup["LAI Diff"]/dfup["SPEI Diff"]
dfdn = pd.read_excel(r"E:/LAI4g/python_MG/抵抗力和恢复力/LAI Sig Change2.xlsx", sheet_name="Sig Down")
dfdn["CR"] = dfdn["LAI Diff"]/dfdn["SPEI Diff"]

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
        d = d[["LAI Diff", "SPEI Diff", "Level", "CR"]]
        if i==0:
            d_all = d
        else:
            d_all = pd.concat([d_all, d], axis=0)
    d_all["State"] = [cat]*len(d_all)
    return d_all

#%%
def plot_box(data, title):
    
    fig = plt.figure(1, figsize=(8, 6), dpi=500)
    ax = fig.add_subplot(111)
    
    palette = ["#FD7F1E", "#1681FC"]
    boxprops = dict(edgecolor='k', linewidth=1.5)
    capprops = dict(color='k', linewidth=1.5)
    whiskerprops = dict(color='k', linewidth=1.5)
    medianprops = dict(color='k', linewidth=1.5)
    meanprops = dict(color='k', linewidth=1.5)
    kwargs = dict(width=0.6, showmeans=True, meanprops=meanprops, boxprops=boxprops, capprops=capprops, 
                  whiskerprops=whiskerprops, medianprops=medianprops, meanline=True)
    f = sns.boxplot(ax=ax, data=data, x="Level", y="CR", hue="State", palette=palette,
                hue_order=["Sig Down", "Sig Up"], **kwargs)
    
    ax.tick_params(labelsize=15)
    ax.set_ylim(-0.05, 1)
    # ax.set_ylim(-0.05, 2)
    ax.set_xlabel("SPEI Change", fontsize=15)
    ax.set_xticks(range(0, 11, 2))
    ax.set_xticklabels(xlabel[::2])
    ax.set_ylabel("LAI Change (Units: m2/m2)", fontsize=15)
    ax.legend(fontsize=15, loc="upper right")
    
    ax.set_title(f"{title}", fontsize=20)
    plt.savefig(rf'E:/LAI4g/python_MG/抵抗力和恢复力/JPG/{title}.jpg',
                bbox_inches='tight')
    
    plt.show()
    
    
def plot_bar(data, title):
    fig = plt.figure(1, figsize=(8, 6), dpi=500)
    ax = fig.add_subplot(111)
    
    palette = ["#FD7F1E", "#1681FC"]
    kwargs = dict(edgecolor='white', linewidth=1.5)
    sns.barplot(data=data, x="Level", y="LAI Diff", 
                hue="State", hue_order=["Sig Down", "Sig Up"], palette=palette,
                **kwargs)
    
    ax.tick_params(labelsize=15)
    ax.set_ylim(0, 260)
    # ax.set_ylim(0, 60)
    ax.set_xlabel("SPEI Change", fontsize=15)
    ax.set_xticks(range(0, 11, 2))
    ax.set_xticklabels(xlabel[::2])
    ax.set_ylabel("LAI Change Counts", fontsize=15)
    ax.legend(fontsize=15, loc="upper right")
    
    ax.set_title(f"{title}", fontsize=20)
    plt.savefig(rf'E:/LAI4g/python_MG/抵抗力和恢复力/JPG/{title}.jpg',
                bbox_inches='tight')
    plt.show()



#%%  转折后1999-2020
bin1 = np.arange(0, 2.76, 0.25)
dfup_2_cut = cut(dfup_2, bin1, "Sig Up")  ## 显著上升
dfdn_2_cut = cut(dfdn_2*-1, bin1, "Sig Down")  ## 显著下降


df2_plot = pd.concat([dfup_2_cut, dfdn_2_cut])
df2_plot["CR"] = abs(df2_plot["CR"])
plot_box(df2_plot, title=r"CR 1999-2020")

#%%
df2_group = df2_plot["LAI Diff"].groupby([df2_plot['State'], df2_plot['Level']])
df2_count = df2_group.count()
df2_count = df2_count.reset_index()
df2_count.sort_values(by=["State", "Level"])

plot_bar(df2_count, title=r"CR 1999-2020 Sample Counts")



#%%  转折前1982-1998
# bin1 = np.arange(0, 2.76, 0.25)
# dfup_1_cut = cut(dfup_1, bin1, "Sig Up")  ## 显著上升
# dfdn_1_cut = cut(dfdn_1*-1, bin1, "Sig Down")  ## 显著下降


# df1_plot = pd.concat([dfup_1_cut, dfdn_1_cut])
# df1_plot["CR"] = abs(df1_plot["CR"])
# plot_box(df1_plot, title=r"CR 1982-1998")

# #%%
# df1_group = df1_plot["LAI Diff"].groupby([df1_plot['State'], df1_plot['Level']])
# df1_count = df1_group.count()
# df1_count = df1_count.reset_index()
# df1_count.sort_values(by=["State", "Level"])

# plot_bar(df1_count, title=r"CR 1982-1998 Sample Counts")