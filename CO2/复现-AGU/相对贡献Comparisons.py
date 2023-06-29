# %%
import netCDF4 as nc
import numpy as np
import pandas as pd

import Senplot as sp

# %%
_ = np.load("./Contribution.npz")
vpd, tmp, co2 = _["vpd"], _["tmp"], _["co2"]

# %%
VPDw = vpd/tmp
VPDc = vpd/co2
VPDwc = vpd/(tmp+co2)

var = ["VPDw", "VPDc", "VPDwc"]
# %%


def read_plot(data1, var):
    print(var)

    print('5 percentile is: ', np.nanpercentile(data1, 5))
    print('95 percentile is: ', np.nanpercentile(data1, 95), "\n")

    levels = [0.25, 0.50, 0.75]
    labels = [">75", "50 - 75", "25 - 75", "<25"]
    counts = sp.cut2(data1)
    labeltitle = f"Percentage"

    if var == 1:
        title = "VPD induced decrease vs. warming induced increase in GOSIF"
    elif var == 2:
        title = "VPD induced decrease vs. co2 induced increase in GOSIF"
    elif var == 3:
        title = "VPD induced decrease vs. warming+co2 induced increase in GOSIF"

    lat = np.linspace(-89.75, 89.75, 360)
    lon = np.linspace(-179.75, 179.75, 720)

    colors = ["#E69800", "#FE0000", "#A80020", "#720000"]
    sp.CreatMap(data1, lon, lat, levels, colors,
                labels, labeltitle, counts, title)

# %%
for i, x in enumerate(var):
    exec(f"read_plot({x}, i+1)")
# %%
