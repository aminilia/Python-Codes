# %% 
########################################################
## Program to download NDBC data from multiple years####
## to a dataframe per station                       ####
########################################################
#Amin Ilia#
#2/22/2024#

import numpy as np
import pandas as pd
import netCDF4 as nc
import xarray as xr

from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.tri as tri

import plotly.graph_objects as go
import requests
import os 
from bs4 import BeautifulSoup

# %% 

yr=[2018,2019,2020]

os.chdir(r'C:\\work\\BOEM\\Task 4\\NDBC')
stc=pd.read_csv('station_code.csv')

# %% 

def ndbc_to_df(stcode,years):
    df=[]
    for j in years:
        dfs=pd.read_csv('https://www.ndbc.noaa.gov/view_text_file.php?filename='+str(stcode)+'h'+str(j)+'.txt.gz&dir=data/historical/stdmet/', delim_whitespace=True)
        dfs.drop([0],axis=0,inplace=True)
        df.append(dfs)
    f_con=pd.concat(df,ignore_index=True)
#    f_con=f_con.apply(pd.to_numeric)
    columns_mapping={'#YY' : 'Year' , 'MM' : 'Month' , 'DD' :'Day', 'hh':'Hour', 'mm':'Minute'}
    f_con.rename(columns=columns_mapping, inplace=True)
    f_con.index=pd.to_datetime(f_con[['Year','Month','Day','Hour','Minute']])
    f_con.drop(['Year','Month','Day','Hour','Minute'],axis=1,inplace=True)
    return f_con

# %% 

df_44095=ndbc_to_df(stc['name'][0],yr)

df_44086=ndbc_to_df(stc['name'][1],yr)

df_44100=ndbc_to_df(stc['name'][2],yr)

df_44088=ndbc_to_df(stc['name'][3],[yr[0],yr[2]])

df_44014=ndbc_to_df(stc['name'][4],yr)

df_44099=ndbc_to_df(stc['name'][5],yr)

df_44089=ndbc_to_df(stc['name'][6],yr)

df_44009=ndbc_to_df(stc['name'][7],yr)

df_44091=ndbc_to_df(stc['name'][8],yr)

df_44065=ndbc_to_df(stc['name'][9],yr)

df_44025=ndbc_to_df(stc['name'][10],yr)

df_44066=ndbc_to_df(stc['name'][11],yr)


# %% 

df_44099=ndbc_to_df(stc['name'][5],yr)

dfss={'df_44095': df_44095, 'df_44086':df_44086, 'df_44100':df_44100,'df_44088':df_44088,'df_44014':df_44014,'df_44099':df_44099, \
             'df_44089':df_44089, 'df_44009':df_44009, 'df_44091':df_44091,'df_44065':df_44065,'df_44025':df_44025, 'df_44066':df_44066}
os.chdir(r'C:\\work\\BOEM\\Task 4\\Validation\\NDBC')
with open('ndbc_df.pck','wb') as file:
    pd.to_pickle(dfss, file)

# %%
