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

dfup = pd.read_excel(
    r"E:/LAI4g/python_MG/抵抗力和恢复力1-2/Sample.xlsx", sheet_name="Sig Up")
dfdn = pd.read_excel(
    r"E:/LAI4g/python_MG/抵抗力和恢复力1-2/Sample.xlsx", sheet_name="Sig Down")

df_num = pd.read_excel(r"E:/LAI4g/python_MG/抵抗力和恢复力1-2/Number of Samples.xlsx")

# %%

dfup_1 = dfup[dfup['Year'] < 2000]
dfup_2 = dfup[dfup['Year'] >= 2000]

dfdn_1 = dfdn[dfdn['Year'] < 2000]
dfdn_2 = dfdn[dfdn['Year'] >= 2000]

# %%


def Xlabel():
    age = np.random.randint((10), size=(10))
    bin = np.arange(1, 2.76, 0.25)
    cats = pd.cut(age, bin, right=False)

    x = cats.categories
    return x


xlabel = Xlabel()

# %%


def cut(data, bin, cat):
    non_corr = data["LAI Diff"]*data["SPEI Diff"]
    non_count = non_corr[non_corr < 0]
    data["non_corr"] = non_corr
    data["non_corr"][data["non_corr"] <= 0] = np.nan
    data = data.dropna()

    b = data.sort_values(by=[f"SPEI Diff"])
    for i in range(len(bin)-1):
        c = b[np.logical_and(b[f"SPEI Diff"] > bin[i],
                             b[f"SPEI Diff"] < bin[i+1])]
        c["Level"] = eval(f'["{str(i).zfill(2)}"]*len(c)')
        d = c.copy()
        d = d[["LAI Diff", "SPEI Diff", "Level"]]
        if i == 0:
            d_all = d
        else:
            d_all = pd.concat([d_all, d], axis=0)
    d_all["State"] = [cat]*len(d_all)
    return d_all

# %%


def ttest(df1, df2):
    marker = []
    level = [str(x).zfill(2) for x in range(7)]
    for lev in level:
        a = df1["LAI Diff"][df1["Level"] == lev]
        b = df2["LAI Diff"][df2["Level"] == lev]
        t, p = stats.ttest_ind(a, b)
        print(lev, p)
        if p > 0.05 and p <= 0.1:
            marker.append("k")
        elif p > 0.01 and p <= 0.05:
            marker.append("b")
        elif p <= 0.01:
            marker.append("r")
        else:
            marker.append('none')
    return marker


def WRStest(df1, df2):
    marker = []
    level = [str(x).zfill(2) for x in range(7)]
    for lev in level:
        a = df1["LAI Diff"][df1["Level"] == lev]
        b = df2["LAI Diff"][df2["Level"] == lev]
        t, p = stats.ranksums(a, b)
        print(lev, p)
        if p > 0.05 and p <= 0.1:
            marker.append("k")
        elif p > 0.01 and p <= 0.05:
            marker.append("b")
        elif p <= 0.01:
            marker.append("r")
        else:
            marker.append('none')
    return marker

# %%


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
                    hue_order=["Drought", "Resilience"], **kwargs)

    x = [str(x).zfill(2) for x in range(7)]
    y = np.zeros(7)
    ax.scatter(x, y-0.02, marker="*", c=mark_test, s=40)
    ax.scatter(x, y-0.07, marker="D", c=markWRS, s=20)

    ax.tick_params(labelsize=15)
    ax.set_ylim(-0.1, 1)
    # ax.set_ylim(-0.1, 0.6)
    ax.set_xlabel("SPEI Absolute Change", fontsize=15)

    ax.set_xticklabels(xlabel)
    ax.set_ylabel("LAI Absolute Change (Units: m${^2}$/m${^2}$)", fontsize=15)
    ax.legend(fontsize=15, loc="upper right")
    ax.text(-0.3, 0.95, "(a)", fontsize=15)

    # ax.set_title(f"{title}", fontsize=20)
    plt.savefig(rf'E:/LAI4g/python_MG/抵抗力和恢复力1-2/JPG/{title}.jpg',
                bbox_inches='tight')

    plt.show()


def plot_bar(data, data2, title):
    fig = plt.figure(1, figsize=(8, 6), dpi=500)
    ax = fig.add_subplot(111)

    palette = ["#FD7F1E", "#1681FC"]
    palette2 = ["lightgray", "lightgray"]
    kwargs = dict(edgecolor='white', linewidth=1.5)
    g = sns.barplot(data=data, x="Level", y="LAI Diff",
                    hue="State", hue_order=["Drought", "Resilience"], palette=palette,
                    **kwargs)
    
    x = []
    for index, row1 in data[:].iterrows():
        x.append(row1.name)
        
    for i, (index, row1) in enumerate(data2[:7].iterrows()):
        text = f"{row1[1]*100:.0f}%"
        g.text(x[i]-0.2, row1[0], text, ha="center", fontsize=15)
        
    for i, (index, row1) in enumerate(data2[7:].iterrows()):
        text = f"{row1[1]*100:.0f}%"
        g.text(x[i]+0.2, row1[0], text, ha="center", fontsize=15)


    ax.tick_params(labelsize=15)
    ax.set_ylim(0, 400)
    # ax.set_ylim(0, 60)
    ax.set_xlabel("SPEI Absolute Change", fontsize=15)

    ax.set_xticklabels(xlabel)
    ax.set_ylabel("Number of Samples", fontsize=15)
    ax.legend(fontsize=15, loc="upper right")
    
    ax.text(-0.4, 380, "(b)", fontsize=15)
    # ax.set_title(f"{title}", fontsize=20)
    plt.savefig(rf'E:/LAI4g/python_MG/抵抗力和恢复力1-2/JPG/{title}.jpg',
                bbox_inches='tight')
    plt.show()


# #%%  转折后2000-2020
# bin1 = np.arange(1, 2.76, 0.25)
# dfup_2_cut = cut(dfup_2, bin1, "Resilience")  ## 显著上升
# dfdn_2_cut = cut(dfdn_2*-1, bin1, "Drought")  ## 显著下降
# mark_test = ttest(dfup_2_cut, dfdn_2_cut)
# markWRS = WRStest(dfup_2_cut, dfdn_2_cut)

# df2_plot = pd.concat([dfup_2_cut, dfdn_2_cut])
# plot_box(df2_plot, title=r"2000-2020 Boxplot")

# #%%
# df2_group = df2_plot["LAI Diff"].groupby([df2_plot['State'], df2_plot['Level']])
# df2_count = df2_group.count()
# df2_count = df2_count.reset_index()
# df2_count.sort_values(by=["State", "Level"])
# df2_count2 = df_num[["2000-2020", "Percentage2"]]

# plot_bar(df2_count, df2_count2, title=r"2000-2020 Sample Counts")


# %%  转折前1982-1999
bin1 = np.arange(1, 2.76, 0.25)
dfup_1_cut = cut(dfup_1, bin1, "Resilience")  # 显著上升
dfdn_1_cut = cut(dfdn_1*-1, bin1, "Drought")  # 显著下降
mark_test = ttest(dfup_1_cut, dfdn_1_cut)
markWRS = WRStest(dfup_1_cut, dfdn_1_cut)

df1_plot = pd.concat([dfup_1_cut, dfdn_1_cut])
plot_box(df1_plot, title=r"1982-1999 Boxplot")

# # %%
# df1_group = df1_plot["LAI Diff"].groupby(
#     [df1_plot['State'], df1_plot['Level']])
# df1_count = df1_group.count()
# df1_count = df1_count.reset_index()
# df1_count.sort_values(by=["State", "Level"])
# df1_count2 = df_num[["1982-1999", "Percentage1"]]

# plot_bar(df1_count, df1_count2, title=r"1982-1999 Sample Counts")

