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

ncfile=nc.Dataset('UV_his_barotropic.nc')

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
fig.show()

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

# %% 

print('Corr9.6=',merged_df['9.6'].corr(merged_df['9.6o']))
print('Corr19.6=',merged_df['19.6'].corr(merged_df['19.6o']))
print('Corr33.6=',merged_df['33.6'].corr(merged_df['33.6o']))

print('std9.6=',merged_df['9.6'].std())
print('std19.6=',merged_df['19.6'].std())
print('std33.6=',merged_df['33.6'].std())

print('std9.6o=',merged_df['9.6o'].std())
print('std19.6o=',merged_df['19.6o'].std())
print('std33.6o=',merged_df['33.6o'].std())

print('bias9.6=',np.mean(merged_df['9.6']-merged_df['9.6o']))
print('bias19.6=',np.mean(merged_df['19.6']-merged_df['19.6o']))
print('bias33.6=',np.mean(merged_df['33.6']-merged_df['33.6o']))

print('rms9.6=',np.sqrt(np.sum((merged_df['9.6']-merged_df['9.6o'])**2)/len(merged_df['9.6'])))
print('rms19.6=',np.sqrt(np.sum((merged_df['19.6']-merged_df['19.6o'])**2)/len(merged_df['19.6'])))
print('rms33.6=',np.sqrt(np.sum((merged_df['33.6']-merged_df['33.6o'])**2)/len(merged_df['33.6'])))

# %% 

comp1=go.Scatter(x=merged_df.index , y=merged_df['9.6o'], mode='markers', name='observation_9.6m' )
comp2=go.Scatter(x=merged_df.index , y=merged_df['9.6'], mode='lines', name='model_9.6m' )

layout= go.Layout(title='9.6m comparison',
                  xaxis=dict(title='Date'),
                  yaxis=dict(title='Current Speed (m/s)'),
                  width=800,
                  height=400)

fig=go.Figure(data=[comp1,comp2], layout=layout)

fig.show()

# %% scatter plots 

comp1=go.Scatter(x=merged_df.index , y=merged_df['33.6o'], mode='markers', name='observation_33.6m' )
comp2=go.Scatter(x=merged_df.index , y=merged_df['33.6'], mode='lines', name='model_33.6m' )

layout= go.Layout(title='33.6m comparison',
                  xaxis=dict(title='Date'),
                  yaxis=dict(title='Current Speed (m/s)'),
                  width=800,
                  height=400)

fig=go.Figure(data=[comp1,comp2], layout=layout)

fig.show()

# %% 

bins = np.arange(0., 0.7, 0.1)

fig, ax=plt.subplots(subplot_kw={'projection':'windrose'})
ax.bar(merged_dfdr['9.6o'], merged_df['9.6o'], bins=bins, normed=True, edgecolor='white', cmap=cm.viridis)
ax.set_legend()
plt.suptitle('Current Rose Observation 9.6m')
plt.savefig('Current_Rose_9.6_obs.png')

fig, ax=plt.subplots(subplot_kw={'projection':'windrose'})
ax.bar(merged_dfdr['9.6'], merged_df['9.6'], bins=bins, normed=True, edgecolor='white', cmap=cm.viridis)
ax.set_legend()
plt.suptitle('Current Rose model 9.6m')
plt.savefig('Current_Rose_9.6_model_tropic.png')

fig, ax=plt.subplots(subplot_kw={'projection':'windrose'})
ax.bar(merged_dfdr['19.6o'],merged_df['19.6o'], bins=bins, normed=True, edgecolor='white', cmap=cm.viridis)
ax.set_legend()
plt.suptitle('Current Rose Observation 19.6m')
plt.savefig('Current_Rose_19.6_obs.png')

fig, ax=plt.subplots(subplot_kw={'projection':'windrose'})
ax.bar(merged_dfdr['19.6'],merged_df['19.6'], bins=bins, normed=True, edgecolor='white', cmap=cm.viridis)
ax.set_legend()
plt.suptitle('Current Rose model 19.6m')
plt.savefig('Current_Rose_19.6_model_tropic.png')

fig, ax=plt.subplots(subplot_kw={'projection':'windrose'})
ax.bar(merged_dfdr['33.6o'], merged_df['33.6o'], bins=bins, normed=True, edgecolor='white', cmap=cm.viridis)
ax.set_legend()
plt.suptitle('Current Rose Observation 33.6m')
plt.savefig('Current_Rose_33.6_obs.png')

fig, ax=plt.subplots(subplot_kw={'projection':'windrose'})
ax.bar(merged_dfdr['33.6'], merged_df['33.6'], bins=bins, normed=True, edgecolor='white', cmap=cm.viridis)
ax.set_legend()
plt.suptitle('Current Rose model 33.6m')
plt.savefig('Current_Rose_33.6_model_tropic.png')

# %% Winter

print('Corr9.6_Winter=',merged_df['9.6'].loc[datetime(2018,12,20):datetime(2019,3,20)] \
      .corr(merged_df['9.6o'].loc[datetime(2018,12,20):datetime(2019,3,20)]))
print('Corr19.6_Winter=',merged_df['19.6'].loc[datetime(2018,12,20):datetime(2019,3,20)] \
      .corr(merged_df['19.6o'].loc[datetime(2018,12,20):datetime(2019,3,20)]))
print('Corr33.6_Winter=',merged_df['33.6'].loc[datetime(2018,12,20):datetime(2019,3,20)] \
      .corr(merged_df['33.6o'].loc[datetime(2018,12,20):datetime(2019,3,20)]))

print('std9.6_Winter=',merged_df['9.6'].loc[datetime(2018,12,20):datetime(2019,3,20)].std())
print('std19.6_Winter=',merged_df['19.6'].loc[datetime(2018,12,20):datetime(2019,3,20)].std())
print('std33.6_Winter=',merged_df['33.6'].loc[datetime(2018,12,20):datetime(2019,3,20)].std())

print('std9.6o_Winter=',merged_df['9.6o'].loc[datetime(2018,12,20):datetime(2019,3,20)].std())
print('std19.6o_Winter=',merged_df['19.6o'].loc[datetime(2018,12,20):datetime(2019,3,20)].std())
print('std33.6o_Winter=',merged_df['33.6o'].loc[datetime(2018,12,20):datetime(2019,3,20)].std())

print('bias9.6_Winter=',np.mean((merged_df['9.6'].loc[datetime(2018,12,20):datetime(2019,3,20)]) \
      -(merged_df['9.6o'].loc[datetime(2018,12,20):datetime(2019,3,20)])))
print('bias9.6_Winter=',np.mean((merged_df['19.6'].loc[datetime(2018,12,20):datetime(2019,3,20)]) \
      -(merged_df['19.6o'].loc[datetime(2018,12,20):datetime(2019,3,20)])))
print('bias33.6_Winter=',np.mean((merged_df['33.6'].loc[datetime(2018,12,20):datetime(2019,3,20)]) \
      -(merged_df['33.6o'].loc[datetime(2018,12,20):datetime(2019,3,20)])))

print('rms9.6_Winter=',np.sqrt(np.sum(((merged_df['9.6'].loc[datetime(2018,12,20):datetime(2019,3,20)]) \
      -(merged_df['9.6o'].loc[datetime(2018,12,20):datetime(2019,3,20)]))**2)/len(merged_df['9.6'].loc[datetime(2018,12,20):datetime(2019,3,20)])))
print('rms19.6_Winter=',np.sqrt(np.sum(((merged_df['19.6'].loc[datetime(2018,12,20):datetime(2019,3,20)]) \
      -(merged_df['19.6o'].loc[datetime(2018,12,20):datetime(2019,3,20)]))**2)/len(merged_df['19.6'].loc[datetime(2018,12,20):datetime(2019,3,20)])))
print('rms33.6_Winter=',np.sqrt(np.sum(((merged_df['33.6'].loc[datetime(2018,12,20):datetime(2019,3,20)]) \
      -(merged_df['33.6o'].loc[datetime(2018,12,20):datetime(2019,3,20)]))**2)/len(merged_df['33.6'].loc[datetime(2018,12,20):datetime(2019,3,20)])))


# %% Spring

print('Corr9.6_Spring=',merged_df['9.6'].loc[datetime(2019,3,20):datetime(2019,6,20)] \
      .corr(merged_df['9.6o'].loc[datetime(2019,3,20):datetime(2019,6,20)]))
print('Corr19.6_Spring=',merged_df['19.6'].loc[datetime(2019,3,20):datetime(2019,6,20)] \
      .corr(merged_df['19.6o'].loc[datetime(2019,3,20):datetime(2019,6,20)]))
print('Corr33.6_Spring=',merged_df['33.6'].loc[datetime(2019,3,20):datetime(2019,6,20)] \
      .corr(merged_df['33.6o'].loc[datetime(2019,3,20):datetime(2019,6,20)]))

print('std9.6_Spring=',merged_df['9.6'].loc[datetime(2019,3,20):datetime(2019,6,20)].std())
print('std19.6_Spring=',merged_df['19.6'].loc[datetime(2019,3,20):datetime(2019,6,20)].std())
print('std33.6_Spring=',merged_df['33.6'].loc[datetime(2019,3,20):datetime(2019,6,20)].std())

print('std9.6o_Spring=',merged_df['9.6o'].loc[datetime(2019,3,20):datetime(2019,6,20)].std())
print('std19.6o_Spring=',merged_df['19.6o'].loc[datetime(2019,3,20):datetime(2019,6,20)].std())
print('std33.6o_Spring=',merged_df['33.6o'].loc[datetime(2019,3,20):datetime(2019,6,20)].std())

print('bias9.6_Spring=',np.mean((merged_df['9.6'].loc[datetime(2019,3,20):datetime(2019,6,20)]) \
      -(merged_df['9.6o'].loc[datetime(2019,3,20):datetime(2019,6,20)])))
print('bias9.6_Spring=',np.mean((merged_df['19.6'].loc[datetime(2019,3,20):datetime(2019,6,20)]) \
      -(merged_df['19.6o'].loc[datetime(2019,3,20):datetime(2019,6,20)])))
print('bias33.6_Spring=',np.mean((merged_df['33.6'].loc[datetime(2019,3,20):datetime(2019,6,20)]) \
      -(merged_df['33.6o'].loc[datetime(2019,3,20):datetime(2019,6,20)])))

print('rms9.6_Spring=',np.sqrt(np.sum(((merged_df['9.6'].loc[datetime(2019,3,20):datetime(2019,6,20)]) \
      -(merged_df['9.6o'].loc[datetime(2019,3,20):datetime(2019,6,20)]))**2)/len(merged_df['9.6'].loc[datetime(2019,3,20):datetime(2019,6,20)])))
print('rms19.6_Spring=',np.sqrt(np.sum(((merged_df['19.6'].loc[datetime(2019,3,20):datetime(2019,6,20)]) \
      -(merged_df['19.6o'].loc[datetime(2019,3,20):datetime(2019,6,20)]))**2)/len(merged_df['19.6'].loc[datetime(2019,3,20):datetime(2019,6,20)])))
print('rms33.6_Spring=',np.sqrt(np.sum(((merged_df['33.6'].loc[datetime(2019,3,20):datetime(2019,6,20)]) \
      -(merged_df['33.6o'].loc[datetime(2019,3,20):datetime(2019,6,20)]))**2)/len(merged_df['33.6'].loc[datetime(2019,3,20):datetime(2019,6,20)])))

# %% Summer

print('Corr9.6_Summer=',merged_df['9.6'].loc[datetime(2019,6,20):datetime(2019,9,20)] \
      .corr(merged_df['9.6o'].loc[datetime(2019,6,20):datetime(2019,9,20)]))
print('Corr19.6_Summer=',merged_df['19.6'].loc[datetime(2019,6,20):datetime(2019,9,20)] \
      .corr(merged_df['19.6o'].loc[datetime(2019,6,20):datetime(2019,9,20)]))
print('Corr33.6_Summer=',merged_df['33.6'].loc[datetime(2019,6,20):datetime(2019,9,20)] \
      .corr(merged_df['33.6o'].loc[datetime(2019,6,20):datetime(2019,9,20)]))

print('std9.6_Summer=',merged_df['9.6'].loc[datetime(2019,6,20):datetime(2019,9,20)].std())
print('std19.6_Summer=',merged_df['19.6'].loc[datetime(2019,6,20):datetime(2019,9,20)].std())
print('std33.6_Summer=',merged_df['33.6'].loc[datetime(2019,6,20):datetime(2019,9,20)].std())

print('std9.6o_Summer=',merged_df['9.6o'].loc[datetime(2019,6,20):datetime(2019,9,20)].std())
print('std19.6o_Summer=',merged_df['19.6o'].loc[datetime(2019,6,20):datetime(2019,9,20)].std())
print('std33.6o_Summer=',merged_df['33.6o'].loc[datetime(2019,6,20):datetime(2019,9,20)].std())

print('bias9.6_Summer=',np.mean((merged_df['9.6'].loc[datetime(2019,6,20):datetime(2019,9,20)]) \
      -(merged_df['9.6o'].loc[datetime(2019,6,20):datetime(2019,9,20)])))
print('bias9.6_Summer=',np.mean((merged_df['19.6'].loc[datetime(2019,6,20):datetime(2019,9,20)]) \
      -(merged_df['19.6o'].loc[datetime(2019,6,20):datetime(2019,9,20)])))
print('bias33.6_Summer=',np.mean((merged_df['33.6'].loc[datetime(2019,6,20):datetime(2019,9,20)]) \
      -(merged_df['33.6o'].loc[datetime(2019,6,20):datetime(2019,9,20)])))

print('rms9.6_Summer=',np.sqrt(np.sum(((merged_df['9.6'].loc[datetime(2019,6,20):datetime(2019,9,20)]) \
      -(merged_df['9.6o'].loc[datetime(2019,6,20):datetime(2019,9,20)]))**2)/len(merged_df['9.6'].loc[datetime(2019,6,20):datetime(2019,9,20)])))
print('rms19.6_Summer=',np.sqrt(np.sum(((merged_df['19.6'].loc[datetime(2019,6,20):datetime(2019,9,20)]) \
      -(merged_df['19.6o'].loc[datetime(2019,6,20):datetime(2019,9,20)]))**2)/len(merged_df['19.6'].loc[datetime(2019,6,20):datetime(2019,9,20)])))
print('rms33.6_Summer=',np.sqrt(np.sum(((merged_df['33.6'].loc[datetime(2019,6,20):datetime(2019,9,20)]) \
      -(merged_df['33.6o'].loc[datetime(2019,6,20):datetime(2019,9,20)]))**2)/len(merged_df['33.6'].loc[datetime(2019,6,20):datetime(2019,9,20)])))


# %% Fall

print('Corr9.6_Fall=',merged_df['9.6'].loc[datetime(2019,9,20):datetime(2019,12,20)] \
      .corr(merged_df['9.6o'].loc[datetime(2019,9,20):datetime(2019,12,20)]))
print('Corr19.6_Fall=',merged_df['19.6'].loc[datetime(2019,9,20):datetime(2019,12,20)] \
      .corr(merged_df['19.6o'].loc[datetime(2019,9,20):datetime(2019,12,20)]))
print('Corr33.6_Fall=',merged_df['33.6'].loc[datetime(2019,9,20):datetime(2019,12,20)] \
      .corr(merged_df['33.6o'].loc[datetime(2019,9,20):datetime(2019,12,20)]))

print('std9.6_Fall=',merged_df['9.6'].loc[datetime(2019,9,20):datetime(2019,12,20)].std())
print('std19.6_Fall=',merged_df['19.6'].loc[datetime(2019,9,20):datetime(2019,12,20)].std())
print('std33.6_Fall=',merged_df['33.6'].loc[datetime(2019,9,20):datetime(2019,12,20)].std())

print('std9.6o_Fall=',merged_df['9.6o'].loc[datetime(2019,9,20):datetime(2019,12,20)].std())
print('std19.6o_Fall=',merged_df['19.6o'].loc[datetime(2019,9,20):datetime(2019,12,20)].std())
print('std33.6o_Fall=',merged_df['33.6o'].loc[datetime(2019,9,20):datetime(2019,12,20)].std())

print('bias9.6_Fall=',np.mean((merged_df['9.6'].loc[datetime(2019,9,20):datetime(2019,12,20)]) \
      -(merged_df['9.6o'].loc[datetime(2019,9,20):datetime(2019,12,20)])))
print('bias9.6_Fall=',np.mean((merged_df['19.6'].loc[datetime(2019,9,20):datetime(2019,12,20)]) \
      -(merged_df['19.6o'].loc[datetime(2019,9,20):datetime(2019,12,20)])))
print('bias33.6_Fall=',np.mean((merged_df['33.6'].loc[datetime(2019,9,20):datetime(2019,12,20)]) \
      -(merged_df['33.6o'].loc[datetime(2019,9,20):datetime(2019,12,20)])))

print('rms9.6_Fall=',np.sqrt(np.sum(((merged_df['9.6'].loc[datetime(2019,9,20):datetime(2019,12,20)]) \
      -(merged_df['9.6o'].loc[datetime(2019,9,20):datetime(2019,12,20)]))**2)/len(merged_df['9.6'].loc[datetime(2019,9,20):datetime(2019,12,20)])))
print('rms19.6_Fall=',np.sqrt(np.sum(((merged_df['19.6'].loc[datetime(2019,9,20):datetime(2019,12,20)]) \
      -(merged_df['19.6o'].loc[datetime(2019,9,20):datetime(2019,12,20)]))**2)/len(merged_df['19.6'].loc[datetime(2019,9,20):datetime(2019,12,20)])))
print('rms33.6_Fall=',np.sqrt(np.sum(((merged_df['33.6'].loc[datetime(2019,9,20):datetime(2019,12,20)]) \
      -(merged_df['33.6o'].loc[datetime(2019,9,20):datetime(2019,12,20)]))**2)/len(merged_df['33.6'].loc[datetime(2019,9,20):datetime(2019,12,20)])))

# %% 

btb9_6=merged_df['9.6'].quantile([0.50, 0.75, 0.95, 0.99])
btb19_6=merged_df['19.6'].quantile([0.50, 0.75, 0.95, 0.99])
btb33_6=merged_df['33.6'].quantile([0.50, 0.75, 0.95, 0.99])

# %% 

dfbt=pd.concat([btb9_6,btb19_6,btb33_6],axis=1)
dfbt.to_csv('tropic_Base_percentile.csv',header=True)

# %% 

merged_df.to_csv('tropic_Base_all.csv',header=True)

# %% 