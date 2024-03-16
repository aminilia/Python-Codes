#############################################################
# Developed by Amin Ilia
# program to compare Temperature and Salinity data 
# from mode and compare those to IOOS glider observations
# The program rewten to be efficient as much as possible
# The program read model data in three days chunks
#############################################################

# %% import library 

import os
import pandas as pd
import numpy as np
import netCDF4 as nc
import xarray as xr
import plotly.graph_objects as go
from datetime import datetime, timedelta

# %% 

os.chdir(r'C:\\work\\BOEM\\Task 4\\DOPPIO')

# %% 

ncd=nc.Dataset('DOPPIOobs.nc')

# %%  

val=ncd.variables['value'][:].data
typ=ncd.variables['type'][:].data
prov=ncd.variables['provenance'][:].data
time=ncd.variables['time'][:].data
depth=ncd.variables['depth'][:].data
lat=ncd.variables['latitude'][:].data
lon=ncd.variables['longitude'][:].data

# %% 

p702=np.zeros([153013,6])

p702[:,0]=time[prov==702]
p702[:,2]=lat[prov==702]
p702[:,1]=lon[prov==702]
p702[:,3]=depth[prov==702]
p702[:,4]=typ[prov==702]
p702[:,5]=val[prov==702]

# %% 
 
df=pd.DataFrame(p702 , columns=['time','lon','lat','depth','type','Val'])

df.index=pd.to_datetime(df['time'] , unit='s')
df['time']=pd.to_datetime(df.index)

dft=df[df['type']==6.0]
dfs=df[df['type']==7.0]
print(dft.index.unique().equals(dft.index.unique()))


dtimeu=dft['time'].unique()
mlonu=dft.groupby(dft.index)['lon'].mean()   # for ech profile
mlatu=dft.groupby(dft.index)['lat'].mean()

# %% 

main_path='F:\\BOEM\\V12\\output_baroclinic_wl_smg0_5'
os.chdir(main_path)
folder_list=os.listdir()
out_folder_list=[]

for i in range(0,len(folder_list)):
    if (folder_list[i]).startswith('output')==True:
        out_folder_list.append(folder_list[i])

# %% 
        
i=0
file_path=main_path+'/'+out_folder_list[1]+'/FlowFM_merged_map.nc'
ncm=nc.Dataset(file_path)

# %% 
sig=ncm.variables['mesh2d_sigmazdepth'][:].data
sigz=ncm.variables['mesh2d_layer_sigma_z'][:].data

wl=ncm.variables['mesh2d_s1'][:].data
wdep=ncm.variables['mesh2d_waterdepth'][:].data

xm=ncm.variables['mesh2d_face_x'][:].data
ym=ncm.variables['mesh2d_face_y'][:].data
zm=ncm.variables['mesh2d_layer_sigma_z'][:].data

# %% 

from scipy.spatial import cKDTree
import numpy as np

coordinates=np.array(list(zip(ym,xm)))
kdtree=cKDTree(coordinates)
xy=np.array([mlatu,mlonu]).T
dist,index=kdtree.query(xy,k=1)

# %% 

dtimeu=dtimeu[dtimeu<np.datetime64('2019-02-01')]    ##################

# %% 

tbase=np.datetime64('2018-01-01')
mbase=np.datetime64('2017-11-01')

dtm=3600*3

gdep=np.array(dft['depth'][dft.index==dtimeu[5]])

ni=0
time=dtimeu[5]

i=-1 

dft['model']=0
dfs['model']=0

for time in dtimeu:
    i=i+1

    difdays=(time-tbase).astype('timedelta64[D]').astype(float)

    ni=int(difdays/3)

    mbtime=ni*np.array(3).astype('timedelta64[D]')+tbase
    dsecb=(time-mbtime).astype('timedelta64[s]').astype(float)

    ist=round(dsecb/dtm)
    
    file_path=main_path+'/'+out_folder_list[ni]+'/FlowFM_merged_map.nc'
    ncmm=nc.Dataset(file_path)
    temp=ncmm.variables['mesh2d_tem1'][ist,index[i],:].data
    sal=ncmm.variables['mesh2d_sa1'][ist,index[i],:].data
    mtime=ncmm.variables['time'][ist].data

    dft['model'][dft.index==time]=np.interp(dft['depth'][dft.index==time], zm, temp)
    dfs['model'][dfs.index==time]=np.interp(dfs['depth'][dfs.index==time], zm, sal)

# %% 

dfs=dfs[dfs['model']!=0]
dft=dft[dft['model']!=0]

stat_s=pd.DataFrame(columns=['Depth','Correlation','Index of Agreement', 'Bias', 'RMSE', 'STD'], index=[(np.linspace(0,12,13))])
stat_s['Depth'].iloc[1:13]=gdep
stat_s['Depth'].iloc[0]=1000

stat_t=pd.DataFrame(columns=['Depth','Correlation','Index of Agreement', 'Bias', 'RMSE', 'STD'], index=[(np.linspace(0,12,13))])
stat_t['Depth'].iloc[1:13]=gdep
stat_t['Depth'].iloc[0]=1000

corrs=dfs.corr()
corrt=dft.corr()
stat_s['Correlation'].iloc[0]=corrs['Val']['model']
stat_t['Correlation'].iloc[0]=corrt['Val']['model']

stat_s['Bias'].iloc[0]=np.mean(dfs['model']-dfs['Val'])
stat_t['Bias'].iloc[0]=np.mean(dft['model']-dft['Val'])

stat_s['STD'].iloc[0]=np.std(dfs['model'])
stat_t['STD'].iloc[0]=np.std(dft['model'])

stat_s['RMSE'].iloc[0]=np.sqrt(np.sum((dfs['model']-dfs['Val'])**2)/len(dfs))
stat_t['RMSE'].iloc[0]=np.sqrt(np.sum((dft['model']-dft['Val'])**2)/len(dft))

stat_s['Index of Agreement'].iloc[0]=1-np.sum((dfs['model']-dfs['Val'])**2)/np.sum((dfs['model']-dfs['model'].mean())**2+(dfs['Val']-dfs['Val'].mean())**2)
stat_t['Index of Agreement'].iloc[0]=1-np.sum((dft['model']-dft['Val'])**2)/np.sum((dft['model']-dft['model'].mean())**2+(dft['Val']-dft['Val'].mean())**2)

# %%

def calrms(group):
    diffval=(group['Val']-group['model'])
    group=np.sqrt(np.mean(diffval**2))
    return group


def calIA(group):
    valmean=group['Val'].mean()
    modelmean=group['model'].mean()
    ia=1-(sum((group['Val']-group['model'])**2)/sum((group['Val']-valmean)**2+(group['model']-modelmean)**2))
    return ia

# %% 

stat_s['Bias'].iloc[1:13]=dfs.groupby('depth').mean()['Val']-dfs.groupby('depth').mean()['model']
stat_s['STD'].iloc[1:13]=dfs.groupby('depth').std()['model']
stat_s['RMSE'].iloc[1:13]=dfs.groupby('depth').apply(calrms)
stat_s['Index of Agreement'].iloc[1:13]=dfs.groupby('depth').apply(calIA)
stat_s['Correlation'].iloc[1:13]=dfs.groupby('depth').apply(lambda x: x['Val'].corr(x['model']))

# %% 

stat_t['Bias'].iloc[1:13]=dft.groupby('depth').mean()['Val']-dft.groupby('depth').mean()['model']
stat_t['STD'].iloc[1:13]=dft.groupby('depth').std()['model']
stat_t['RMSE'].iloc[1:13]=dft.groupby('depth').apply(calrms)
stat_t['Index of Agreement'].iloc[1:13]=dft.groupby('depth').apply(calIA)
stat_t['Correlation'].iloc[1:13]=dft.groupby('depth').apply(lambda x: x['Val'].corr(x['model']))

# %%

os.chdir('E:\\BOEM\\validation')

stat_s.to_csv('Stats_Sal_glider_baroclinic_wl_smg0_5.csv')

stat_t.to_csv('Stats_Temp_glider_baroclinic_wl_smg0_5.csv')

# %% 

