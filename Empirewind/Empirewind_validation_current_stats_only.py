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

import os

# %% 

model_id='06'

os.chdir(r'C:\\work\\BOEM\\Task 4\\Validation\\NDBC\\his')
ncfile=nc.Dataset('Comb_his_WL_sm1_0_sd15.nc')

stname=ncfile['station_name'][:].data
stname=stname.astype('str')
merged_stname=[''.join(row) for row in stname]

# %% 

print(merged_stname[16])

# %% 

time = np.asarray(ncfile.variables['time'])
uu = np.asarray(ncfile.variables['x_velocity'][:,16,:])
vv = np.asarray(ncfile.variables['y_velocity'][:,16,:])

xx=np.asarray(ncfile.variables['station_geom_node_coordx'][16])
yy=np.asarray(ncfile.variables['station_geom_node_coordy'][16])
zz=np.asarray(ncfile.variables['zcoordinate_c'][:,16,:])

# %% 


# %%

u_plot=go.Heatmap(z=uu[:,25:39].T, colorscale='Viridis')
layout=go.Layout(
    title='u z',
    xaxis=dict(title='Time'),
    yaxis=dict(title='Depth'))
fig=go.Figure(data=[u_plot], layout=layout)
fig.show()

v_plot=go.Heatmap(z=vv[:,25:39].T, colorscale='Viridis')
layout=go.Layout(
    title='v z',
    xaxis=dict(title='Time'),
    yaxis=dict(title='Depth'))
fig=go.Figure(data=[v_plot], layout=layout)
#fig.show()

# %% 

ign=436

from scipy.interpolate import interp1d
res_u=np.zeros([np.size(uu,0)-ign,3])
res_v=np.zeros([np.size(uu,0)-ign,3])
for i in range(0,np.size(uu,0)-ign):
    interfunction_u=interp1d(zz[i,25:39],uu[i,25:39], kind='linear')
    res_u[i,:]=interfunction_u([-9.6,-19.6,-33.6])

    interfunction_v=interp1d(zz[i,25:39],vv[i,25:39], kind='linear')
    res_v[i,:]=interfunction_v([-9.6,-19.6,-33.6])

# %% 
    
def convert_2_datetime(seconds):
    time_delta=seconds.astype('timedelta64[s]')
    stime=np.datetime64('2017-11-01T00:00:00')
    res_dtime=stime+time_delta
    return res_dtime

# %% 

dtime=convert_2_datetime(time[:-ign])
uv=np.sqrt(res_u**2+res_v**2)

dfuv=pd.DataFrame(uv,columns=['9.6','19.6','33.6'])
dfuv.index=pd.to_datetime(dtime)

# %% 

dfdr = (90-np.degrees(np.arctan2(res_v,res_u)))  % 360

dfdr=pd.DataFrame(dfdr,columns=['9.6','19.6','33.6'])
dfdr.index=pd.to_datetime(dtime)

# %% import obs

dfobs=pd.read_csv('C:\work\BOEM\Task 4\EmpireWind_Data\Empire_Current_Mooring.csv',header=[0])  
dfobs.drop(['SeaWaterTemp','SeaWaterTempQual','WaterConduct','WaterConductQual','AbsolutePressure','AbsolutePressureQual'], axis=1,inplace=True)
dfobs.dropna(inplace=True)

dfobs.index=pd.to_datetime(dfobs['time'],format='%Y-%m-%dT%H:%M:%SZ')   # if there is a two header it cannot convert
dfobs.drop(['time','station_name','latitude','longitude'], axis=1, inplace=True)

# %% 

dfobs['CurDir_rad']=dfobs['CurDirn']/180*np.pi

dfobs['CurSpd_u']=dfobs['CurSpd']*np.cos((90-dfobs['CurDirn'])/180*np.pi)
dfobs['CurSpd_v']=dfobs['CurSpd']*np.sin((90-dfobs['CurDirn'])/180*np.pi)

# %% calculate hourly average for observation data

dfsp_con=pd.concat([dfobs['CurSpd'].loc[dfobs['depth']==9.6], \
                dfobs['CurSpd'].loc[dfobs['depth']==19.6], \
                dfobs['CurSpd'].loc[dfobs['depth']==33.6]],axis=1, \
                     keys=['9.6o','19.6o','33.6o'] )


dfdr_con=pd.concat([dfobs['CurDirn'].loc[dfobs['depth']==9.6], \
                dfobs['CurDirn'].loc[dfobs['depth']==19.6], \
                dfobs['CurDirn'].loc[dfobs['depth']==33.6]], \
                axis=1, keys=['9.6o','19.6o','33.6o'] )

# %% 

dfsp=dfsp_con.rolling(window=60,min_periods=1).mean()  # calculate hourly 

# %% 

merged_df=pd.merge(dfsp,dfuv,left_index=True,right_index=True,how='inner')

merged_dfdr=pd.merge(dfdr_con,dfdr,left_index=True,right_index=True,how='inner')

merged_df.dropna(inplace=True)

# %% 

stat=pd.DataFrame(columns=['Correlation','Index of Agreement', 'Bias', 'PercentBias' , 'RMSE', 'STD', 'STDo'], index=[(np.linspace(0,2,3))])

# %% 
stat['Correlation'].iloc[0]=merged_df['9.6'].corr(merged_df['9.6o'])
stat['Correlation'].iloc[1]=merged_df['19.6'].corr(merged_df['19.6o'])
stat['Correlation'].iloc[2]=merged_df['33.6'].corr(merged_df['33.6o'])

stat['STD'].iloc[0]=merged_df['9.6'].std()
stat['STD'].iloc[1]=merged_df['19.6'].std()
stat['STD'].iloc[2]=merged_df['33.6'].std()

stat['STDo'].iloc[0]=merged_df['9.6o'].std()
stat['STDo'].iloc[1]=merged_df['19.6o'].std()
stat['STDo'].iloc[2]=merged_df['33.6o'].std()

stat['Bias'].iloc[0]=np.mean(merged_df['9.6']-merged_df['9.6o'])
stat['Bias'].iloc[1]=np.mean(merged_df['19.6']-merged_df['19.6o'])
stat['Bias'].iloc[2]=np.mean(merged_df['33.6']-merged_df['33.6o'])

stat['RMSE'].iloc[0]=np.sqrt(np.sum((merged_df['9.6']-merged_df['9.6o'])**2)/len(merged_df['9.6']))
stat['RMSE'].iloc[1]=np.sqrt(np.sum((merged_df['19.6']-merged_df['19.6o'])**2)/len(merged_df['19.6']))
stat['RMSE'].iloc[2]=np.sqrt(np.sum((merged_df['33.6']-merged_df['33.6o'])**2)/len(merged_df['33.6']))

stat['PercentBias'].iloc[0]=stat['Bias'].iloc[0]/np.mean(merged_df['9.6o'])*100
stat['PercentBias'].iloc[1]=stat['Bias'].iloc[1]/np.mean(merged_df['19.6o'])*100
stat['PercentBias'].iloc[2]=stat['Bias'].iloc[2]/np.mean(merged_df['33.6o'])*100

# %% 

os.chdir(r'C:\\work\\BOEM\\Task 4\\Validation\\EmpireWind')
statm=stat.mean()
statm.to_csv('stats_mean_ID'+model_id+'.csv')

# %% 

