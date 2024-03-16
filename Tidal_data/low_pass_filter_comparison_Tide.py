"""
Created on Mon Jan 17 2024
Program to cal residual by a lowpass filter
@author: Amin.Ilia
"""
### this program use low pass filter to remove tide from observations###
# %%

import numpy as np 
import pandas as pd 
import xarray as xr 
import  netCDF4 as nc 
from datetime import datetime, timedelta
import pyTMD as tmd
import pickle
import matplotlib.pylab as plt
from scipy import signal

# %% 

timedel=pd.date_range("2023-09-01", "2024-04-01", freq="30min").values
    
with open('data_obs.pickle' , 'rb') as f:
    lev_obs=pickle.load(f)

top=pd.read_csv('C:\work\CSA\FES2014\program\\2023_Obs\CSA_TPXO9_Gauge.csv',names=['time','level'], header=None)

top.index=pd.to_datetime(top['time'])
top.drop('time',axis=1,inplace=True)
top.columns=['Level']

# %%

fes=pd.DataFrame(lev_obs.data.T)
fes.index=pd.to_datetime(timedel)
fes.columns=['Level']
# dfl.drop(columns=[5,6], inplace=True) #remove NAN columns

#######################load observations and plot them!!!!#####################################

# %% 

obs=pd.read_csv('C:\work\CSA\FES2014\program\\2023_Obs\92139_Sept-Dec 2023_Tides.csv')

obs.index=pd.to_datetime(obs['Time'])+timedelta(hours=5)

# %% estimate residual

b,a=signal.butter(4,1/(60*6),'low')
fgust = signal.filtfilt(b, a, obs['Level'], method="gust")

res_obs=obs['Level']-fgust

# %% 

res_obs=res_obs.loc[datetime(2023,9,6,14,30,0):datetime(2023,12,14,23,30,0)]
hlfh=pd.date_range(start='2023-09-06 14:30:00',end='2023-12-14 23:45:00',freq='30T')
res_obs=res_obs.loc[res_obs.index.isin(hlfh)]

top=top.loc[datetime(2023,9,6,14,30,0):datetime(2023,12,14,23,45,0)]

fes=fes.loc[datetime(2023,9,6,14,30,0):datetime(2023,12,14,23,45,0)]

# %% 

fes['Level']=fes['Level']-fes['Level'].mean()
top['Level']=top['Level']-top['Level'].mean()

# %% 

print('correlation_with_FES2014=',fes.corrwith(res_obs))
print('correlation_with_TOPEX=',top.corrwith(res_obs))

print('RMS_FES2014=',((fes['Level']-res_obs)**2).mean()**0.5)
print('RMS_TOPEX=',((top['Level']-res_obs)**2).mean()**0.5)

print('BIAS_FES2014=',(fes['Level']-res_obs).mean())
print('BIAS_TOPEX=',(top['Level']-res_obs).mean())

# %% 

fig, ax=plt.subplots()
ax.plot( fes['Level'].loc[datetime(2023,11,17):datetime(2023,12,14)] , 'b', label='FES2014')
ax.plot( top['Level'].loc[datetime(2023,11,17):datetime(2023,12,14)] , 'r', label='Topex')
ax.plot( res_obs.loc[datetime(2023,11,17):datetime(2023,12,14)] , 'k', label='Tide Gauge')
ax.legend()
fig.autofmt_xdate(rotation=35)
plt.suptitle('Comparison of Topex, FES2014, and Obsevation')
plt.savefig('FES2014_TOPEX_obs_Comparison_GMT_res.png')

# %% 
