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
from scipy import stats
import seaborn as sns
import xarray as xr

plt.rcParams['font.sans-serif'] = ['times new roman']  # 指定默认字体
warnings.filterwarnings("ignore")
# %%

df = pd.read_excel("Sample2.xlsx")

# %%

df1 = df[df['YEAR'] < 2000]
df2 = df[df['YEAR'] >= 2000]

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
    data.columns = ["YEAR", "SPEI Diff", "LAI Diff"]
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
def ttest(df1, df2):
    marker = []
    level = [str(x).zfill(2) for x in range(11)]
    for lev in level:
        a = df1["LAI Diff"][df1["Level"]==lev]
        b = df2["LAI Diff"][df2["Level"]==lev]
        t, p = stats.ttest_ind(a, b)
        print(lev, p)
        if p>0.05 and p<=0.1:
            marker.append("k")
        elif p>0.01 and p<=0.05:
            marker.append("b")
        elif p<=0.01:
            marker.append("r")
        else:
            marker.append('none')
    print("")
    return marker
        
def WRStest(df1, df2):
    marker = []
    level = [str(x).zfill(2) for x in range(11)]
    for lev in level:
        a = df1["LAI Diff"][df1["Level"]==lev]
        b = df2["LAI Diff"][df2["Level"]==lev]
        t, p = stats.ranksums(a, b)
        print(lev, p)
        if p>0.05 and p<=0.1:
            marker.append("k")
        elif p>0.01 and p<=0.05:
            marker.append("b")
        elif p<=0.01:
            marker.append("r")
        else:
            marker.append('none')
    print("")
    return marker

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
    f = sns.boxplot(ax=ax, data=data, x="Level", y="LAI Diff", hue="State", palette=palette,
                hue_order=["Immediate", "Resilience"], **kwargs)
    
    x = [str(x).zfill(2) for x in range(11)]
    y = np.zeros(11)-0.025
    ax.scatter(x, y, marker="*", c=mark_test, s=40)
    ax.scatter(x, y-0.025, marker="D", c=markWRS, s=20)
    
    ax.tick_params(labelsize=15)
    ax.set_ylim(-0.1, 1.3)
    # ax.set_ylim(-0.1, 0.5)
    ax.set_xlabel("SPEI Change", fontsize=15)
    ax.set_xticks(range(0, 11, 2))
    ax.set_xticklabels(xlabel[::2])
    ax.set_ylabel("LAI Change (Units: m2/m2)", fontsize=15)
    ax.legend(fontsize=15, loc="upper right")
    
    ax.set_title(f"{title}", fontsize=20)
    # plt.savefig(rf'E:/LAI4g/python_MG/抵抗力和恢复力2/JPG/{title}.jpg',
    #             bbox_inches='tight')
    
    plt.show()
    
    
def plot_bar(data, title):
    fig = plt.figure(1, figsize=(8, 6), dpi=500)
    ax = fig.add_subplot(111)
    
    palette = ["#FD7F1E", "#1681FC"]
    kwargs = dict(edgecolor='white', linewidth=1.5)
    sns.barplot(data=data, x="Level", y="LAI Diff", 
                hue="State", hue_order=["Immediate", "Resilience"], palette=palette,
                **kwargs)
    
    ax.tick_params(labelsize=15)
    ax.set_ylim(0, 350)
    # ax.set_ylim(0, 60)
    ax.set_xlabel("SPEI Change", fontsize=15)
    ax.set_xticks(range(0, 11, 2))
    ax.set_xticklabels(xlabel[::2])
    ax.set_ylabel("LAI Change Counts", fontsize=15)
    ax.legend(fontsize=15, loc="upper right")
    
    ax.set_title(f"{title}", fontsize=20)
    # plt.savefig(rf'E:/LAI4g/python_MG/抵抗力和恢复力2/JPG/{title}.jpg',
    #             bbox_inches='tight')
    plt.show()



#%%  转折后1999-2020
bin1 = np.arange(0, 2.76, 0.25)

df2_up = df2.iloc[:, [0, 2, 4]]
df2_up = cut(df2_up, bin1, "Resilience")  ## 显著上升

df2_dn = df2.iloc[:, [0, 1, 3]]
df2_dn = cut(df2_dn*-1, bin1, "Immediate")  ## 显著下降
mark_test = ttest(df2_up, df2_dn)
markWRS = WRStest(df2_up, df2_dn)

df2_plot = pd.concat([df2_up, df2_dn])
plot_box(df2_plot, title=r"2000-2020")

#%%
df2_group = df2_plot["LAI Diff"].groupby([df2_plot['State'], df2_plot['Level']])
df2_count = df2_group.count()
df2_count = df2_count.reset_index()
df2_count.sort_values(by=["State", "Level"])

plot_bar(df2_count, title=r"2000-2020 Sample Counts")



# #%%  转折前1982-1998
# bin1 = np.arange(0, 2.76, 0.25)

# df1_up = df1.iloc[:, [0, 2, 4]]
# df1_up = cut(df1_up, bin1, "Resilience")  ## 显著上升

# df1_dn = df1.iloc[:, [0, 1, 3]]
# df1_dn = cut(df1_dn*-1, bin1, "Immediate")  ## 显著下降
# mark_test = ttest(df1_up, df1_dn)
# markWRS = WRStest(df1_up, df1_dn)

# df1_plot = pd.concat([df1_up, df1_dn])
# plot_box(df1_plot, title=r"1982-1999")

# # #%%
# df1_group = df1_plot["LAI Diff"].groupby([df1_plot['State'], df1_plot['Level']])
# df1_count = df1_group.count()
# df1_count = df1_count.reset_index()
# df1_count.sort_values(by=["State", "Level"])

# plot_bar(df1_count, title=r"1982-1999 Sample Counts")


