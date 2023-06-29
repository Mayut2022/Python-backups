# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 21:09:30 2023

转折前后同干旱和湿润对比

@author: MaYutong
"""


import warnings
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

dfup = pd.read_excel(r"E:/LAI4g/python_MG/抵抗力和恢复力1-2/Sample.xlsx", sheet_name="Sig Up")
dfdn = pd.read_excel(r"E:/LAI4g/python_MG/抵抗力和恢复力1-2/Sample.xlsx", sheet_name="Sig Down")

df_num = pd.read_excel(r"E:/LAI4g/python_MG/抵抗力和恢复力1-2/Number of Samples.xlsx")
# %%

dfup_1 = dfup[dfup['Year'] < 2000]
dfup_2 = dfup[dfup['Year'] >= 2000]

dfdn_1 = dfdn[dfdn['Year'] < 2000]
dfdn_2 = dfdn[dfdn['Year'] >= 2000]


#%%

# for i in range(1, 4):
#     print(f"dfup_{i}")
#     exec(f"print(dfup_{i}.max())")
#     exec(f"print(dfup_{i}.min())")
#     print("")
    
# for i in range(1, 4):
#     print(f"dfdn_{i}")
#     exec(f"print(dfdn_{i}.max())")
#     exec(f"print(dfdn_{i}.min())")
#     print("")

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

#%%
def Xlabel():
    age = np.random.randint((10), size=(10))
    # bin = np.arange(1, 2.76, 0.25)
    bin = np.arange(-2.75, -0.9, 0.25)
    cats = pd.cut(age, bin, right=False)
    
    x = cats.categories
    return x[::-1]

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
    
    fig = plt.figure(1, figsize=(8, 6), dpi=500)
    ax = fig.add_subplot(111)
    
    # palette = ["#1681FC", "#FD7F1E"]
    palette = ["moccasin", "coral"]
    # palette = ["lightskyblue", "dodgerblue"]
    # flierprops = dict(marker='.')
    boxprops = dict(edgecolor='k', linewidth=1.5)
    capprops = dict(color='k', linewidth=1.5)
    whiskerprops = dict(color='k', linewidth=1.5)
    medianprops = dict(color='k', linewidth=1.5)
    meanprops = dict(color='k', linewidth=1.5)
    kwargs = dict(width=0.6, showmeans=True, meanprops=meanprops, boxprops=boxprops, capprops=capprops, 
                  whiskerprops=whiskerprops, medianprops=medianprops, meanline=True)
    f = sns.boxplot(ax=ax, data=data, x="Level", y="LAI Diff", hue="State", palette=palette,
                **kwargs)
    
    x = [str(x).zfill(2) for x in range(7)]
    y = np.zeros(7)
    ax.scatter(x, y-0.02, marker="*", c=mark_test, s=40)
    ax.scatter(x, y-0.07, marker="D", c=markWRS, s=20)
    
    ax.tick_params(labelsize=13)
    ax.set_ylim(-0.1, 1)
    yticks = np.arange(0, 1.1, 0.2)
    ax.set_yticks(yticks)
    
    yticks2 = [0.0, -0.2, -0.4, -0.6, -0.8, -1.0]
    ax.set_yticklabels(yticks2)
    ax.set_xlabel("SPEI Change", fontsize=15)
    ax.set_xticklabels(xlabel)
    ax.set_ylabel("LAI Change (Units: m2/m2)", fontsize=15)
    
    ax.text(-0.3, 0.95, "(a)", fontsize=15)
    ax.legend(fontsize=15, loc="upper right")
    # ax.set_title(f"{title}", fontsize=20)
    plt.savefig(rf'E:/LAI4g/python_MG/抵抗力和恢复力1-2/JPG/{title}.jpg',
                bbox_inches='tight')
    
    plt.show()
    
    
def plot_bar(data, data2, title):
    fig = plt.figure(1, figsize=(8, 6), dpi=500)
    ax = fig.add_subplot(111)
    
    # palette = ["#1681FC", "#FD7F1E"]
    palette = ["moccasin", "coral"]
    # palette = ["lightskyblue", "dodgerblue"]
    kwargs = dict(edgecolor='white', linewidth=1.5)
    g = sns.barplot(data=data, x="Level", y="LAI Diff", 
                hue="State", hue_order=["1982-1999", "2000-2020"], palette=palette,
                **kwargs)
    
    x = []
    for index, row1 in data[:].iterrows():
        x.append(row1.name)
        
    for i, (index, row1) in enumerate(data2[:].iterrows()):
        text1 = f"{row1[1]*100:.0f}%"
        text2 = f"{row1[3]*100:.0f}%"
        g.text(x[i]-0.2, row1[0], text1, ha="center", fontsize=15)
        g.text(x[i]+0.2, row1[2], text2, ha="center", fontsize=15)
        
    
    ax.tick_params(labelsize=13)
    ax.set_ylim(0, 400)
    ax.set_xlabel("SPEI Change", fontsize=15)
    ax.set_xticklabels(xlabel)
    ax.set_ylabel("Number of Samples", fontsize=15)
    
    ax.text(-0.4, 380, "(b)", fontsize=15)
    ax.legend(fontsize=15, loc="upper right")
    # ax.set_title(f"{title}", fontsize=20)
    plt.savefig(rf'E:/LAI4g/python_MG/抵抗力和恢复力1-2/JPG/{title}.jpg',
                bbox_inches='tight')
    plt.show()



#%%  干旱/显著下降对比
bin1 = np.arange(1, 2.76, 0.25)
dfdn_1_cut = cut(dfdn_1*-1, bin1, "1982-1999")  

dfdn_2_cut = cut(dfdn_2*-1, bin1, "2000-2020")  

mark_test = ttest(dfdn_1_cut, dfdn_2_cut)
markWRS = WRStest(dfdn_1_cut, dfdn_2_cut)

dfdn_plot = pd.concat([dfdn_1_cut, dfdn_2_cut])
plot_box(dfdn_plot, title=r"Drought Comparison")

# #%%
dfdn_group = dfdn_plot["LAI Diff"].groupby([dfdn_plot['State'], dfdn_plot['Level']])
dfdn_count = dfdn_group.count()
dfdn_count = dfdn_count.reset_index()
dfdn_count.sort_values(by=["State", "Level"])
dfdn_count2 = df_num.loc[:6, ["1982-1999", "Percentage1", "2000-2020", "Percentage2"]]

plot_bar(dfdn_count, dfdn_count2, title=r"Drought Comparison Sample Counts")



# # #%%  湿润/显著上升对比
# bin1 = np.arange(1, 2.76, 0.25)
# dfup_1_cut = cut(dfup_1, bin1, "1982-1999")  

# dfup_2_cut = cut(dfup_2, bin1, "2000-2020")  

# mark_test = ttest(dfup_1_cut, dfup_2_cut)
# markWRS = WRStest(dfup_1_cut, dfup_2_cut)

# dfup_plot = pd.concat([dfup_1_cut, dfup_2_cut])
# plot_box(dfup_plot, title=r"Resilience Comparison")

# #%%
# dfup_group = dfup_plot["LAI Diff"].groupby([dfup_plot['State'], dfup_plot['Level']])
# dfup_count = dfup_group.count()
# dfup_count = dfup_count.reset_index()
# dfup_count.sort_values(by=["State", "Level"])
# dfup_count2 = df_num.loc[7:, ["1982-1999", "Percentage1", "2000-2020", "Percentage2"]]

# plot_bar(dfup_count, dfup_count2, title=r"Resilience Comparison Sample Counts")



