# -*- coding: utf-8 -*-
"""
Created on Mon May 22 10:53:55 2023

@author: MaYutong
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
from scipy import stats
import seaborn as sns
import xarray as xr

plt.rcParams['font.sans-serif'] = ['times new roman']  # 指定默认字体
warnings.filterwarnings("ignore")
#%%


dfup = pd.read_excel(r"E:/LAI4g/python_MG/抵抗力和恢复力1-2/Sample.xlsx", sheet_name="Sig Up")
dfdn = pd.read_excel(r"E:/LAI4g/python_MG/抵抗力和恢复力1-2/Sample.xlsx", sheet_name="Sig Down")

dfup_1 = dfup[dfup['Year'] < 2000]
dfup_2 = dfup[dfup['Year'] >= 2000]

dfdn_1 = dfdn[dfdn['Year'] < 2000]
dfdn_2 = dfdn[dfdn['Year'] >= 2000]

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


#%% 干旱转折前
bin1 = np.arange(1, 2.76, 0.25)
dfup_1_cut = cut(dfup_1, bin1, "Resilience")  ## 显著上升
dfdn_1_cut = cut(dfdn_1*-1, bin1, "Drought")  ## 显著下降
df1_plot = pd.concat([dfup_1_cut, dfdn_1_cut])

df1_group = df1_plot["LAI Diff"].groupby([df1_plot['State'], df1_plot['Level']])
df1_count = df1_group.count()
df1_count = df1_count.reset_index()
df1_count.sort_values(by=["State", "Level"])


#%% 干旱转折后
bin1 = np.arange(1, 2.76, 0.25)
dfup_2_cut = cut(dfup_2, bin1, "Resilience")  ## 显著上升
dfdn_2_cut = cut(dfdn_2*-1, bin1, "Drought")  ## 显著下降
df2_plot = pd.concat([dfup_2_cut, dfdn_2_cut])

df2_group = df2_plot["LAI Diff"].groupby([df2_plot['State'], df2_plot['Level']])
df2_count = df2_group.count()
df2_count = df2_count.reset_index()
df2_count.sort_values(by=["State", "Level"])



#%%
DFup = pd.read_excel(r"E:/LAI4g/python_MG/抵抗力和恢复力1-2/Sample2.xlsx", sheet_name="Sig Up")
DFdn = pd.read_excel(r"E:/LAI4g/python_MG/抵抗力和恢复力1-2/Sample2.xlsx", sheet_name="Sig Down")

DFup_1 = DFup[DFup['Year'] < 2000]
DFup_2 = DFup[DFup['Year'] >= 2000]

DFdn_1 = DFdn[DFdn['Year'] < 2000]
DFdn_2 = DFdn[DFdn['Year'] >= 2000]


#%% 干旱转折前
bin1 = np.arange(1, 2.76, 0.25)
DFup_1_cut = cut(DFup_1, bin1, "Resilience")  ## 显著上升
DFdn_1_cut = cut(DFdn_1*-1, bin1, "Drought")  ## 显著下降
DF1_plot = pd.concat([DFup_1_cut, DFdn_1_cut])

DF1_group = DF1_plot["LAI Diff"].groupby([DF1_plot['State'], DF1_plot['Level']])
DF1_count = DF1_group.count()
DF1_count = DF1_count.reset_index()
DF1_count.sort_values(by=["State", "Level"])


#%% 干旱转折后
bin1 = np.arange(1, 2.76, 0.25)
DFup_2_cut = cut(DFup_2, bin1, "Resilience")  ## 显著上升
DFdn_2_cut = cut(DFdn_2*-1, bin1, "Drought")  ## 显著下降
DF2_plot = pd.concat([DFup_2_cut, DFdn_2_cut])

DF2_group = DF2_plot["LAI Diff"].groupby([DF2_plot['State'], DF2_plot['Level']])
DF2_count = DF2_group.count()
DF2_count = DF2_count.reset_index()
DF2_count.sort_values(by=["State", "Level"])


#%%
df_output = df1_count.copy() 
df_output["LAI Diff2"] = DF1_count["LAI Diff"]
df_output["Percentage1"] = df_output["LAI Diff2"]/df_output["LAI Diff"]
df_output.rename(columns={'LAI Diff': "1982-1999", 'LAI Diff2': "1982-1999 2"}, inplace=True)

df_output["2000-2020"] = df2_count["LAI Diff"]
df_output["2000-2020 2"] = DF2_count["LAI Diff"]
df_output["Percentage2"] = df_output["2000-2020 2"]/df_output["2000-2020"]


#%%
df_output.to_excel("Number of Samples.xlsx", engine='xlsxwriter', index=False)