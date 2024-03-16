# %%

# comparison with observation at 

import numpy as np 
import pandas as pd 
import xarray as xr 
import  netCDF4 as nc 
from datetime import datetime, timedelta
import pyTMD as tmd

import pickle

import matplotlib.pylab as plt

# %% 
fespath='C:\work\CSA\FES2014\program\\'

# %% 

#deltat=timedelta(minutes=30)
#times= [datetime(2023,3,1,0,0)+i*deltat for i in range(17520)]

ref=datetime(2000,1,1,0,0,0,0)
stime=datetime(2023,9,1,0,0,0)
etime=datetime(2024,4,1,0,0,0)
times=[(etime-ref).days*24*3600,(stime-ref).days*24*3600]

#timedays=[(dtime-ref).seconds for dtime in times]
timedel=pd.date_range("2023-09-01", "2024-04-01", freq="30min").values

# %% 

coor=np.array([10.30,-61.75])   # trhe point is outside fes domin choosed one nearest

coor[1]=360+coor[1]

lev_obs=np.zeros([10225])

# %% 

lev_obs=tmd.compute_tide_corrections(coor[1], coor[0], timedel, DIRECTORY=fespath,
    MODEL='FES2014', EPSG=4326, EPOCH=(2000,1,1,0,0,0), TYPE='time series', TIME='datetime',
    METHOD='spline', FILL_VALUE=np.nan)

# %% 

with open('data_obs.pickle', 'wb') as f:
    pickle.dump(lev_obs, f)

# %% 
    
with open('data_obs.pickle' , 'rb') as f:
    lev_obs=pickle.load(f)

top=pd.read_csv('C:\work\CSA\FES2014\program\\2023_Obs\CSA_TPXO9_Gauge.csv',names=['time','level'], header=None)

top.index=pd.to_datetime(top['time'])

# %%

fes=pd.DataFrame(lev_obs.data.T)
fes.index=pd.to_datetime(timedel)

# dfl.drop(columns=[5,6], inplace=True) #remove NAN columns


#######################load observations and plot them!!!!#####################################

# %% 

obs=pd.read_csv('C:\work\CSA\FES2014\program\\2023_Obs\92139_Sept-Dec 2023_Tides.csv')

obs.index=pd.to_datetime(obs['Time'])+timedelta(hours=5)

# %% 

fig, ax=plt.subplots()
ax.plot( fes[0].loc[datetime(2023,11,15):datetime(2023,11,30)] , 'b', label='FES2014')
ax.plot( top['level'].loc[datetime(2023,11,15):datetime(2023,11,30)] , 'r', label='Topex')
ax.plot( obs['Level'].loc[datetime(2023,11,15):datetime(2023,11,30)] , 'k', label='Tide Gauge')
ax.legend()
fig.autofmt_xdate(rotation=35)
plt.suptitle('Comparison of Topex, FES2014, and Obsevation')
plt.savefig('FES2014_TOPEX_obs_Comparison_GMT.png')

# %% 



