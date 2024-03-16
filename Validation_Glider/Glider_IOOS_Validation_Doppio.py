#############################################################
# Developed by Amin Ilia
# program to compare Temperature and Salinity data 
# from mode and compare those to IOOS glider observations
# The program rewten to be efficient as much as possible
# The program read model data in three days chunks
#############################################################
# Developed by Amin Ilia for glider observation
 
# %% import library 

import os
import pandas as pd
import numpy as np
import netCDF4 as nc
import xarray as xr
from datetime import datetime, timedelta
import calendar
import sys

# %% 

os.chdir(r'E:\\BOEM\\validation\\Data')

model_id='04'
main_path='F:\\BOEM\\v12\\output_spatial_vari_Eddy_sd2_smo1_calib'

# %% 

ncd=nc.Dataset('DOPPIOobs_fd9c_27a6_c5d9.nc')

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
mlonu=dft.groupby(dft.index)['lon'].mean()   # for each profile
mlatu=dft.groupby(dft.index)['lat'].mean()

if len(dtimeu) == len(mlonu):
    print("The sizes are the same.")
else:
    print("The sizes are different.")
    sys.exit()

# %% 

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

start_date = np.datetime64('2018-02-01')
end_date = np.datetime64('2019-02-01')

mlonu = mlonu[(dtimeu > start_date) & (dtimeu < end_date)]
mlatu = mlatu[(dtimeu > start_date) & (dtimeu < end_date)]
dtimeu = dtimeu[(dtimeu > start_date) & (dtimeu < end_date)]

# %% 

from scipy.spatial import cKDTree
import numpy as np

coordinates=np.array(list(zip(ym,xm)))
kdtree=cKDTree(coordinates)
xy=np.array([mlatu,mlonu]).T
dist,index=kdtree.query(xy,k=1)

# %% 

tbase=np.datetime64('2018-01-01')
mbase=np.datetime64('2017-11-01')

dtm=3600*3

gdep=np.array(dft['depth'][dft.index==dtimeu[5]])

dft.loc[:,'model']=np.nan
dfs.loc[:,'model']=np.nan

# %% 
i=-1 
for timei in dtimeu:
    i=i+1

    difdays=(timei-tbase).astype('timedelta64[D]').astype(float)

    ni=int(difdays/3)

    mbtime=ni*np.array(3).astype('timedelta64[D]')+tbase
    dsecb=(timei-mbtime).astype('timedelta64[s]').astype(float)

    ist=round(dsecb/dtm)
    
    file_path=main_path+'/'+out_folder_list[ni]+'/FlowFM_merged_map.nc'
    ncmm=nc.Dataset(file_path)
    temp=ncmm.variables['mesh2d_tem1'][ist,index[i],:].data
    sal=ncmm.variables['mesh2d_sa1'][ist,index[i],:].data
    mtime=ncmm.variables['time'][ist].data

    timei_maskt = dft.index == timei
    depth_values = dft.loc[timei_maskt, 'depth']
    temp_interpolated = np.interp(depth_values, zm, temp)
    dft.loc[timei_maskt, 'model'] = temp_interpolated

    timei_masks = dfs.index == timei
    depth_values = dfs.loc[timei_masks, 'depth']
    sal_interpolated = np.interp(depth_values, zm, sal)
    dfs.loc[timei_masks, 'model'] = sal_interpolated


# %% 

dfs.dropna(inplace=True)
dft.dropna(inplace=True)

stat_s=pd.DataFrame(columns=['Depth','Correlation','Index of Agreement', 'Bias','PercentBias','RMSE','RMSE/STDo', 'STD','STDm/STDo'], index=range(0,13))
stat_s['Depth'].iloc[1:13]=gdep
stat_s['Depth'].iloc[0]=1000

stat_t=pd.DataFrame(columns=['Depth','Correlation','Index of Agreement', 'Bias','PercentBias', 'RMSE','RMSE/STDo', 'STD','STDm/STDo'], index=range(0,13))
stat_t['Depth'].iloc[1:13]=gdep
stat_t['Depth'].iloc[0]=1000

stat_s.index=stat_s.index.astype(int)
stat_t.index=stat_t.index.astype(int)

corrs=dfs.corr()
corrt=dft.corr()
stat_s['Correlation'].iloc[0]=corrs['Val']['model']
stat_t['Correlation'].iloc[0]=corrt['Val']['model']

stat_s['Bias'].iloc[0]=np.mean(dfs['model']-dfs['Val'])
stat_t['Bias'].iloc[0]=np.mean(dft['model']-dft['Val'])

stat_s['PercentBias'].iloc[0]=stat_s['Bias'].iloc[0]/np.mean(dfs['Val'])*100
stat_t['PercentBias'].iloc[0]=stat_t['Bias'].iloc[0]/np.mean(dft['Val'])*100

stat_s['STD'].iloc[0]=np.std(dfs['model'])
stat_t['STD'].iloc[0]=np.std(dft['model'])

stat_s['STDm/STDo'].iloc[0]=np.std(dfs['model'])/np.std(dfs['Val'])
stat_t['STDm/STDo'].iloc[0]=np.std(dft['model'])/np.std(dft['Val'])

stat_s['RMSE'].iloc[0]=np.sqrt(np.sum((dfs['model']-dfs['Val'])**2)/len(dfs))
stat_t['RMSE'].iloc[0]=np.sqrt(np.sum((dft['model']-dft['Val'])**2)/len(dft))

stat_s['RMSE/STDo'].iloc[0]=np.std(dfs['model'])/np.std(dfs['Val'])
stat_t['RMSE/STDo'].iloc[0]=np.std(dft['model'])/np.std(dft['Val'])

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

stat_s['Bias'].iloc[1:13]=dfs.groupby('depth').mean()['model']-dfs.groupby('depth').mean()['Val']
stat_s['PercentBias'].iloc[1:13]=stat_s['Bias'].iloc[1:13]/dfs.groupby('depth').mean()['Val'].values*100
stat_s['STD'].iloc[1:13]=dfs.groupby('depth').std()['model']
stat_s['STDm/STDo'].iloc[1:13]=dfs.groupby(dfs.index.month).std()['model']/dfs.groupby(dfs.index.month).std()['Val']
stat_s['RMSE'].iloc[1:13]=dfs.groupby('depth').apply(calrms)
stat_s['RMSE/STDo'].iloc[1:13]=stat_s['RMSE'].iloc[1:13]/dfs.groupby(dfs.index.month).std()['Val']
stat_s['Index of Agreement'].iloc[1:13]=dfs.groupby('depth').apply(calIA)
stat_s['Correlation'].iloc[1:13]=dfs.groupby('depth').apply(lambda x: x['Val'].corr(x['model']))

# %% 

stat_t['Bias'].iloc[1:13]=dft.groupby('depth').mean()['model']-dft.groupby('depth').mean()['Val']
stat_t['PercentBias'].iloc[1:13]=stat_t['Bias'].iloc[1:13]/dft.groupby('depth').mean()['Val'].values*100
stat_t['STD'].iloc[1:13]=dft.groupby('depth').std()['model']
stat_t['STDm/STDo'].iloc[1:13]=dft.groupby(dft.index.month).std()['model']/dft.groupby(dft.index.month).std()['Val']
stat_t['RMSE'].iloc[1:13]=dft.groupby('depth').apply(calrms)
stat_t['RMSE/STDo'].iloc[1:13]=stat_t['RMSE'].iloc[1:13]/dft.groupby(dft.index.month).std()['Val']
stat_t['Index of Agreement'].iloc[1:13]=dft.groupby('depth').apply(calIA)
stat_t['Correlation'].iloc[1:13]=dft.groupby('depth').apply(lambda x: x['Val'].corr(x['model']))

# %%

os.chdir(r'E:\\BOEM\\validation')

stat_s.to_csv('Stats_Sal_glider_ID'+model_id+'.csv')

stat_t.to_csv('Stats_Temp_glider_ID'+model_id+'.csv')

# %% 

stat_sm=pd.DataFrame(columns=['Month','Correlation','Index of Agreement', 'Bias', 'PercentBias','RMSE', 'STD'], index=range(1,13))
stat_sm['Month']=[calendar.month_name[i] for i in range(1,13)]

stat_sm['Bias']=dfs.groupby(dfs.index.month).mean()['model']-dfs.groupby(dfs.index.month).mean()['Val']
stat_sm['PercentBias']=stat_sm['Bias']/dfs.groupby(dfs.index.month).mean()['Val'].values*100
stat_sm['STD']=dfs.groupby(dfs.index.month).std()['model']
stat_sm['STDm/STDo']=dfs.groupby(dfs.index.month).std()['model']/dfs.groupby(dfs.index.month).std()['Val']
stat_sm['RMSE']=dfs.groupby(dfs.index.month).apply(calrms)
stat_sm['RMSE/STDo']=stat_sm['RMSE']/dfs.groupby(dfs.index.month).std()['Val']
stat_sm['Index of Agreement']=dfs.groupby(dfs.index.month).apply(calIA)
stat_sm['Correlation']=dfs.groupby(dfs.index.month).apply(lambda x: x['Val'].corr(x['model']))

# %% 

stat_tm=pd.DataFrame(columns=['Month','Correlation','Index of Agreement', 'Bias', 'RMSE', 'STD'], index=range(1,13))
stat_tm['Month']=[calendar.month_name[i] for i in range(1,13)]

stat_tm['Bias']=dft.groupby(dft.index.month).mean()['model']-dft.groupby(dft.index.month).mean()['Val']
stat_tm['PercentBias']=stat_tm['Bias']/dft.groupby(dft.index.month).mean()['Val'].values*100
stat_tm['STD']=dft.groupby(dft.index.month).std()['model']
stat_tm['STDm/STDo']=dft.groupby(dft.index.month).std()['model']/dft.groupby(dft.index.month).std()['Val']
stat_tm['RMSE']=dft.groupby(dft.index.month).apply(calrms)
stat_tm['RMSE/STDo']=stat_tm['RMSE']/dft.groupby(dft.index.month).std()['Val']
stat_tm['Index of Agreement']=dft.groupby(dft.index.month).apply(calIA)
stat_tm['Correlation']=dft.groupby(dft.index.month).apply(lambda x: x['Val'].corr(x['model']))

# %% 

stat_sm.to_csv('Stats_Sal_glider_monthly_ID'+model_id+'.csv')

stat_tm.to_csv('Stats_Temp_glider_monthly_ID'+model_id+'.csv')

# %% 

