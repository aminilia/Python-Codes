# %% 

import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc
import plotly.graph_objects as go

from datetime import timedelta,datetime
import matplotlib.pyplot as plt 
import matplotlib.cm as cm

from windrose import WindroseAxes
from windrose import plot_windrose

# %% 

tr_base=pd.read_csv('tropic_base_percentile.csv')
tr_sc3=pd.read_csv('tropic_sc3_percentile.csv')

tr_base_all=pd.read_csv('tropic_all.csv',index_col=0)
tr_sc3_all=pd.read_csv('tropic_sc3_sc3_all.csv',index_col=0)

# %% 

tr_differ=tr_sc3-tr_base
tr_differ.index=tr_base['Unnamed: 0']

# %% 

tr_dif=tr_sc3_all-tr_base_all

# %% 

((tr_sc3_all-tr_base_all)**2).sum(axis=0)/len(tr_sc3_all)

# %% 




# %% 