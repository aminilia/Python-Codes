# %%

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

bn=pd.read_csv('C:\work\CSA\FES2014\Water_Level\Boundary_nodes.csv')

#deltat=timedelta(minutes=30)
#times= [datetime(2023,3,1,0,0)+i*deltat for i in range(17520)]

ref=datetime(2000,1,1,0,0,0,0)
stime=datetime(2023,3,1,0,0,0)
etime=datetime(2024,4,1,0,0,0)
times=[(etime-ref).days*24*3600,(stime-ref).days*24*3600]

#timedays=[(dtime-ref).seconds for dtime in times]
timedel=pd.date_range("2023-03-01", "2024-04-01", freq="30min").values

# %% 

bn['lon']=bn['lon']+360

bnd=bn[bn['Node'].str.endswith('A')]

bnd.reset_index(drop=True, inplace=True)

lev=np.zeros([18,19057])

for index,row in bnd.iterrows():
    
    lev[index,:]=tmd.compute_tide_corrections(row['lon'], row['lat'], timedel, DIRECTORY=fespath,
    MODEL='FES2014', EPSG=4326, EPOCH=(2000,1,1,0,0,0), TYPE='time series', TIME='datetime',
    METHOD='spline', FILL_VALUE=np.nan)

# %% 

with open('data.pickle', 'wb') as f:
    pickle.dump(lev, f)

# %% 
    
with open('data.pickle' , 'rb') as f:
    lev=pickle.load(f)


# %% 

seg14=pd.read_csv(r'C:\work\CSA\FES2014\Water_Level\TPXO_timeseries\CSA_TPXO_BC_North_Seg14_End_A_TPXO.csv')
seg14.index=pd.to_datetime(seg14['time'])

seg3=pd.read_csv(r'C:\work\CSA\FES2014\Water_Level\TPXO_timeseries\CSA_TPXO_BC_South_Seg3_End_A_TPXO.csv')
seg3.index=pd.to_datetime(seg3['time'])

# %%

dfl=pd.DataFrame(lev.T)
dfl.index=pd.to_datetime(timedel)

dfl.drop(columns=[5,6], inplace=True) #remove NAN columns

# %% 

fig, ax=plt.subplots()
ax.plot( dfl[14].loc[datetime(2023,3,15):datetime(2023,3,30)] , label='FES2014')
ax.plot( seg14['level'].loc[datetime(2023,3,15):datetime(2023,3,30)] , label='Topex')
ax.legend()
fig.autofmt_xdate(rotation=35)
plt.suptitle('Comparison of Topex and FES2014_SEG'+str(14))
plt.savefig('FES2014_TOPEX_Comparison'+str(14)+'.png')

# %% 
fig, ax=plt.subplots()
ax.plot( dfl[3].loc[datetime(2023,3,15):datetime(2023,3,30)] , label='FES2014')
ax.plot( seg3['level'].loc[datetime(2023,3,15):datetime(2023,3,30)] , label='Topex')
ax.legend()
fig.autofmt_xdate(rotation=35)
plt.suptitle('Comparison of Topex and FES2014_SEG'+str(3))
plt.savefig('FES2014_TOPEX_Comparison'+str(3)+'.png')

# %% 
