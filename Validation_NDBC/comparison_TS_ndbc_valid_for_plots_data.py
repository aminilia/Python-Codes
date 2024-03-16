# %% 
#Amin Ilia# 
#2/24/2024#

import numpy as np
import pandas as pd
import netCDF4 as nc
import xarray as xr

from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.tri as tri

import plotly.io as pio
import plotly.graph_objects as go
import requests
import os 
from bs4 import BeautifulSoup

from skill_metrics import taylor_diagram


# %% 'df_44086', 'df_44100', 'df_44088', 'df_44014', 'df_44099', 'df_44089', 'df_44009', 'df_44091', 'df_44065', 'df_44025', 'df_44066'
os.chdir('C:\\work\\BOEM\\Task 4\\Validation\\NDBC')
with open('ndbc_df.pck','rb') as file:
    loaded=pd.read_pickle('ndbc_df.pck')

# %% 

def convert_2_datetime(seconds):
    time_delta=seconds.astype('timedelta64[s]')
    stime=np.datetime64('2017-11-01T00:00:00')
    res_dtime=stime+time_delta
    return res_dtime

# %% 
ncfile=nc.Dataset('Comb_his_WL_current.nc')

stname=ncfile['station_name'][:].data
stname=stname.astype('str')
merged_stname=[''.join(row) for row in stname]

time = np.asarray(ncfile.variables['time'])
temp = np.asarray(ncfile.variables['temp'][:,:,:])
sal = np.asarray(ncfile.variables['sal'][:,:,:])

dtime=convert_2_datetime(time)
df_model1=pd.DataFrame(temp[:,:,39])
df_model1.index=pd.to_datetime(dtime)

# %% 

ncfile=nc.Dataset('Comb_his_WL_sm1_0.nc')
temp2 = np.asarray(ncfile.variables['temp'][:,:,:])
time2 = np.asarray(ncfile.variables['time'])
dtime2=convert_2_datetime(time2)
df_model2=pd.DataFrame(temp2[:,:,39])
df_model2.index=pd.to_datetime(dtime2)


ncfile=nc.Dataset('Comb_his_Cu_sm1_0_sd15.nc')
temp3 = np.asarray(ncfile.variables['temp'][:,:,:])
df_model3=pd.DataFrame(temp3[:,:,39])
time3 = np.asarray(ncfile.variables['time'])
dtime3=convert_2_datetime(time3)
df_model3.index=pd.to_datetime(dtime3)

# %% 

print(merged_stname[4])
df_44095=loaded['df_44095']

print(merged_stname[5])
df_44086=loaded['df_44086']

print(merged_stname[6])
df_44100=loaded['df_44100']

print(merged_stname[7])
df_44088=loaded['df_44088']

print(merged_stname[8])
df_44014=loaded['df_44014']

print(merged_stname[9])
df_44099=loaded['df_44099']
df_44099['WTMP']=df_44099['WTMP'].loc[df_44099['WTMP'] != 999.0 ]

print(merged_stname[10])
df_44089=loaded['df_44089']

print(merged_stname[11])
df_44009=loaded['df_44009']

print(merged_stname[12])
df_44091=loaded['df_44091']

print(merged_stname[13])
df_44065=loaded['df_44065']

print(merged_stname[14])
df_44025=loaded['df_44025']

print(merged_stname[15])
df_44066=loaded['df_44066']

# %% 44009

fig=go.Figure()
comp1=fig.add_trace(go.Scatter(x=df_44009.index , y=df_44009['WTMP'], mode='lines', name=merged_stname[11][5:25]+'' ))
comp2=fig.add_trace(go.Scatter(x=df_model1.index , y=df_model1[11], mode='lines', name='WL_Cu_sd4_Smag=1.0'))
comp3=fig.add_trace(go.Scatter(x=df_model2.index , y=df_model2[11], mode='lines', name='WL_sd4_Smag=1.0' ))
comp4=fig.add_trace(go.Scatter(x=df_model3.index , y=df_model3[11], mode='lines', name='WL_Cu_sd15_Smag=1.0' ))

fig.update_xaxes(range=[datetime(2018,1,1),datetime(2019,2,1)])
fig.update_yaxes(range=[0,30])
fig.update_yaxes(type='linear')
fig.update_layout(title='<b>Comparison for '+merged_stname[11][5:10]+'<b>',
                  xaxis=dict(title='<b>Date<b>',tickfont=dict(size=12)),
                  yaxis=dict(title='<b>Temperature [C]<b>',tickfont=dict(size=12)),
                  font=dict(size=12, color='black', family='sans-serif'),
                  width=800,
                  height=400)
fig.show()
plt.savefig(merged_stname[11][5:25]+'comp.png', format='png', dpi=300)


# %% 44100

fig=go.Figure()
comp1=fig.add_trace(go.Scatter(x=df_44100.index , y=df_44100['WTMP'], mode='lines', name=merged_stname[6][5:25]+'' ))
comp2=fig.add_trace(go.Scatter(x=df_model1.index , y=df_model1[6], mode='lines', name='WL_Cu_sd4_Smag=1.0'))
comp3=fig.add_trace(go.Scatter(x=df_model2.index , y=df_model2[6], mode='lines', name='WL_sd4_Smag=1.0' ))
comp4=fig.add_trace(go.Scatter(x=df_model3.index , y=df_model3[6], mode='lines', name='WL_Cu_sd15_Smag=1.0' ))

fig.update_xaxes(range=[datetime(2018,1,1),datetime(2019,2,1)])
fig.update_yaxes(range=[0,30])
fig.update_yaxes(type='linear')
fig.update_layout(title='<b>Comparison for '+merged_stname[6][5:10]+'<b>',
                  xaxis=dict(title='<b>Date<b>',tickfont=dict(size=12)),
                  yaxis=dict(title='<b>Temperature [C]<b>',tickfont=dict(size=12)),
                  font=dict(size=12, color='black', family='sans-serif'),
                  width=800,
                  height=400)
fig.show()
plt.savefig(merged_stname[6][5:25]+'comp.png', format='png', dpi=300)

# %% 44088

fig=go.Figure()
comp1=fig.add_trace(go.Scatter(x=df_44088.index , y=df_44088['WTMP'], mode='lines', name=merged_stname[7][5:25]+'' ))
comp2=fig.add_trace(go.Scatter(x=df_model1.index , y=df_model1[7], mode='lines', name='WL_Cu_sd4_Smag=1.0'))
comp3=fig.add_trace(go.Scatter(x=df_model2.index , y=df_model2[7], mode='lines', name='WL_sd4_Smag=1.0' ))
comp4=fig.add_trace(go.Scatter(x=df_model3.index , y=df_model3[7], mode='lines', name='WL_Cu_sd15_Smag=1.0' ))

fig.update_xaxes(range=[datetime(2018,1,1),datetime(2019,2,1)])
fig.update_yaxes(range=[0,30])
fig.update_yaxes(type='linear')
fig.update_layout(title='<b>Comparison for '+merged_stname[7][5:10]+'<b>',
                  xaxis=dict(title='<b>Date<b>',tickfont=dict(size=12)),
                  yaxis=dict(title='<b>Temperature [C]<b>',tickfont=dict(size=12)),
                  font=dict(size=12, color='black', family='sans-serif'),
                  width=800,
                  height=400)
fig.show()
plt.savefig(merged_stname[7][5:25]+'comp.png', format='png', dpi=300)

# %% 44066

fig=go.Figure()
comp1=fig.add_trace(go.Scatter(x=df_44066.index , y=df_44066['WTMP'], mode='lines', name=merged_stname[15][5:25]+'' ))
comp2=fig.add_trace(go.Scatter(x=df_model1.index , y=df_model1[15], mode='lines', name='WL_Cu_sd4_Smag=1.0'))
comp3=fig.add_trace(go.Scatter(x=df_model2.index , y=df_model2[15], mode='lines', name='WL_sd4_Smag=1.0' ))
comp4=fig.add_trace(go.Scatter(x=df_model3.index , y=df_model3[15], mode='lines', name='WL_Cu_sd15_Smag=1.0' ))

fig.update_xaxes(range=[datetime(2018,1,1),datetime(2019,2,1)])
fig.update_yaxes(range=[0,30])
fig.update_yaxes(type='linear')
fig.update_layout(title='<b>Comparison for '+merged_stname[15][5:10]+'<b>',
                  xaxis=dict(title='<b>Date<b>',tickfont=dict(size=12)),
                  yaxis=dict(title='<b>Temperature [C]<b>',tickfont=dict(size=12)),
                  font=dict(size=12, color='black', family='sans-serif'),
                  width=800,
                  height=400)
fig.show()
plt.savefig(merged_stname[15][5:25]+'comp.png', format='png', dpi=300)

# %% 44014

fig=go.Figure()
comp1=fig.add_trace(go.Scatter(x=df_44014.index , y=df_44014['WTMP'], mode='lines', name=merged_stname[8][5:25]+'' ))
comp2=fig.add_trace(go.Scatter(x=df_model1.index , y=df_model1[8], mode='lines', name='WL_Cu_sd4_Smag=1.0'))
comp3=fig.add_trace(go.Scatter(x=df_model2.index , y=df_model2[8], mode='lines', name='WL_sd4_Smag=1.0' ))
comp4=fig.add_trace(go.Scatter(x=df_model3.index , y=df_model3[8], mode='lines', name='WL_Cu_sd15_Smag=1.0' ))

fig.update_xaxes(range=[datetime(2018,1,1),datetime(2019,2,1)])
fig.update_yaxes(range=[0,30])
fig.update_yaxes(type='linear')
fig.update_layout(title='<b>Comparison for '+merged_stname[8][5:10]+'<b>',
                  xaxis=dict(title='<b>Date<b>',tickfont=dict(size=12)),
                  yaxis=dict(title='<b>Temperature [C]<b>',tickfont=dict(size=12)),
                  font=dict(size=12, color='black', family='sans-serif'),
                  width=800,
                  height=400)
fig.show()
plt.savefig(merged_stname[8][5:25]+'comp.png', format='png', dpi=300)

# %% 44065

fig=go.Figure()
comp1=fig.add_trace(go.Scatter(x=df_44065.index , y=df_44065['WTMP'], mode='lines', name=merged_stname[13][5:25]+'' ))
comp2=fig.add_trace(go.Scatter(x=df_model1.index , y=df_model1[13], mode='lines', name='WL_Cu_sd4_Smag=1.0'))
comp3=fig.add_trace(go.Scatter(x=df_model2.index , y=df_model2[13], mode='lines', name='WL_sd4_Smag=1.0' ))
comp4=fig.add_trace(go.Scatter(x=df_model3.index , y=df_model3[13], mode='lines', name='WL_Cu_sd15_Smag=1.0' ))

fig.update_xaxes(range=[datetime(2018,1,1),datetime(2019,2,1)])
fig.update_yaxes(range=[0,30])
fig.update_yaxes(type='linear')
fig.update_layout(title='<b>Comparison for '+merged_stname[13][5:10]+'<b>',
                  xaxis=dict(title='<b>Date<b>',tickfont=dict(size=12)),
                  yaxis=dict(title='<b>Temperature [C]<b>',tickfont=dict(size=12)),
                  font=dict(size=12, color='black', family='sans-serif'),
                  width=800,
                  height=400)
fig.show()
#plt.savefig(merged_stname[13][5:25]+'comp.png', format='png', dpi=300)
#fig.write_image(merged_stname[13][5:25]+'comp.png', format='png', scale=3) 

# %% 44089

fig=go.Figure()
comp1=fig.add_trace(go.Scatter(x=df_44089.index , y=df_44089['WTMP'], mode='lines', name=merged_stname[10][5:25]+'' ))
comp2=fig.add_trace(go.Scatter(x=df_model1.index , y=df_model1[10], mode='lines', name='WL_Cu_sd4_Smag=1.0'))
comp3=fig.add_trace(go.Scatter(x=df_model2.index , y=df_model2[10], mode='lines', name='WL_sd4_Smag=1.0' ))
comp4=fig.add_trace(go.Scatter(x=df_model3.index , y=df_model3[10], mode='lines', name='WL_Cu_sd15_Smag=1.0' ))

fig.update_xaxes(range=[datetime(2018,1,1),datetime(2019,2,1)])
fig.update_yaxes(range=[0,30])
fig.update_yaxes(type='linear')
fig.update_layout(title='<b>Comparison for '+merged_stname[10][5:10]+'<b>',
                  xaxis=dict(title='<b>Date<b>',tickfont=dict(size=12)),
                  yaxis=dict(title='<b>Temperature [C]<b>',tickfont=dict(size=12)),
                  font=dict(size=12, color='black', family='sans-serif'),
                  width=800,
                  height=400)
fig.show()
plt.savefig(merged_stname[10][5:25]+'comp.png', format='png', dpi=300)

# %% stats 

stat=pd.DataFrame(columns=['St_name','Correlation','Index of Agreement', 'Bias', 'RMSE', 'STD'], index=[(np.linspace(0,11,12))])
stat['St_name']=merged_stname[4:16]

# %% 44095

dfm=pd.DataFrame(df_model1[4])
dfm=dfm.loc[datetime(2018,2,1):datetime(2019,2,1)]
dfm['time']=dfm.index

dfo=pd.DataFrame(pd.to_numeric(df_44095['WTMP']))
dfo=dfo.loc[datetime(2018,2,1):datetime(2019,2,1)]
dfo=dfo['WTMP'].loc[dfo['WTMP'] != 999.0 ]
dfo=pd.DataFrame(dfo)
dfo=dfo.resample('30T').ffill(limit=1).interpolate(method='linear',axis=1) # you cannot increase res of data with this method
dfo.dropna(inplace=True)
dfo['time']=dfo.index

merged=pd.merge(dfo, dfm, on='time',how='outer')
merged.dropna(inplace=True)
merged.index=pd.to_datetime(merged['time'])
merged.drop(['time'],inplace=True,axis=1)

co=merged.corr()
stat['Correlation'].loc[0]=co['WTMP'][4]
stat['Bias'].loc[0]=np.mean(merged['WTMP']-merged[4])
stat['RMSE'].loc[0]=np.sqrt(np.sum((merged['WTMP']-merged[4])**2)/len(merged))
stat['Index of Agreement'].loc[0]=1-np.sum((merged['WTMP']-merged[4])**2)/np.sum((merged['WTMP']-merged['WTMP'].mean())**2+(merged[4]-merged[4].mean())**2)
stat['STD'].loc[0]=np.std(merged[4])

# %% 44086

dfm=pd.DataFrame(df_model1[5])
dfm=dfm.loc[datetime(2018,2,1):datetime(2019,2,1)]
dfm['time']=dfm.index

dfo=pd.DataFrame(pd.to_numeric(df_44086['WTMP']))
dfo.index=pd.to_datetime(dfo.index)
dfo=dfo.loc[datetime(2018,2,1):datetime(2019,2,1)]
dfo=dfo['WTMP'].loc[dfo['WTMP'] != 999.0 ]
dfo=pd.DataFrame(dfo)
dfo=dfo.resample('30T').ffill(limit=1).interpolate(method='linear',axis=1) # you cannot increase res of data with this method
dfo.dropna(inplace=True)
dfo['time']=dfo.index

merged=pd.merge(dfo, dfm, on='time',how='outer')
merged.dropna(inplace=True)
merged.index=pd.to_datetime(merged['time'])
merged.drop(['time'],inplace=True,axis=1)

co=merged.corr()
stat['Correlation'].loc[1]=co['WTMP'][5]
stat['Bias'].loc[1]=np.mean(merged['WTMP']-merged[5])
stat['RMSE'].loc[1]=np.sqrt(np.sum((merged['WTMP']-merged[5])**2)/len(merged))
stat['Index of Agreement'].loc[1]=1-np.sum((merged['WTMP']-merged[5])**2)/np.sum((merged['WTMP']-merged['WTMP'].mean())**2+(merged[5]-merged[5].mean())**2)
stat['STD'].loc[1]=np.std(merged[5])

# %% 44100

dfm=pd.DataFrame(df_model1[6])
dfm=dfm.loc[datetime(2018,2,1):datetime(2019,2,1)]
dfm['time']=dfm.index

dfo=pd.DataFrame(pd.to_numeric(df_44100['WTMP']))
dfo.index=pd.to_datetime(dfo.index)
dfo=dfo.loc[datetime(2018,2,1):datetime(2019,2,1)]
dfo=dfo['WTMP'].loc[dfo['WTMP'] != 999.0 ]
dfo=pd.DataFrame(dfo)
dfo=dfo.resample('30T').ffill(limit=1).interpolate(method='linear',axis=1) # you cannot increase res of data with this method
dfo.dropna(inplace=True)
dfo['time']=dfo.index

merged=pd.merge(dfo, dfm, on='time',how='outer')
merged.dropna(inplace=True)
merged.index=pd.to_datetime(merged['time'])
merged.drop(['time'],inplace=True,axis=1)

co=merged.corr()
stat['Correlation'].loc[2]=co['WTMP'][6]
stat['Bias'].loc[2]=np.mean(merged['WTMP']-merged[6])
stat['RMSE'].loc[2]=np.sqrt(np.sum((merged['WTMP']-merged[6])**2)/len(merged))
stat['Index of Agreement'].loc[2]=1-np.sum((merged['WTMP']-merged[6])**2)/np.sum((merged['WTMP']-merged['WTMP'].mean())**2+(merged[6]-merged[6].mean())**2)
stat['STD'].loc[2]=np.std(merged[6])

# %% 44088

dfm=pd.DataFrame(df_model1[7])
dfm=dfm.loc[datetime(2018,2,1):datetime(2019,2,1)]
dfm['time']=dfm.index

dfo=pd.DataFrame(pd.to_numeric(df_44088['WTMP']))
dfo.index=pd.to_datetime(dfo.index)
dfo=dfo.loc[datetime(2018,2,1):datetime(2019,2,1)]
dfo=dfo['WTMP'].loc[dfo['WTMP'] != 999.0 ]
dfo=pd.DataFrame(dfo)
dfo=dfo.resample('30T').ffill(limit=1).interpolate(method='linear',axis=1) # you cannot increase res of data with this method
dfo.dropna(inplace=True)
dfo['time']=dfo.index

merged=pd.merge(dfo, dfm, on='time',how='outer')
merged.dropna(inplace=True)
merged.index=pd.to_datetime(merged['time'])
merged.drop(['time'],inplace=True,axis=1)

co=merged.corr()
stat['Correlation'].loc[3]=co['WTMP'][7]
stat['Bias'].loc[3]=np.mean(merged['WTMP']-merged[7])
stat['RMSE'].loc[3]=np.sqrt(np.sum((merged['WTMP']-merged[7])**2)/len(merged))
stat['Index of Agreement'].loc[3]=1-np.sum((merged['WTMP']-merged[7])**2)/np.sum((merged['WTMP']-merged['WTMP'].mean())**2+(merged[7]-merged[7].mean())**2)
stat['STD'].loc[3]=np.std(merged[7])

# %% 44014

dfm=pd.DataFrame(df_model1[8])
dfm=dfm.loc[datetime(2018,2,1):datetime(2019,2,1)]
dfm['time']=dfm.index

dfo=pd.DataFrame(pd.to_numeric(df_44014['WTMP']))
dfo.index=pd.to_datetime(dfo.index)
dfo=dfo.loc[datetime(2018,2,1):datetime(2019,2,1)]
dfo=dfo['WTMP'].loc[dfo['WTMP'] != 999.0 ]
dfo=pd.DataFrame(dfo)
dfo=dfo.resample('30T').ffill(limit=1).interpolate(method='linear',axis=1) # you cannot increase res of data with this method
dfo.dropna(inplace=True)
dfo['time']=dfo.index

merged=pd.merge(dfo, dfm, on='time',how='outer')
merged.dropna(inplace=True)
merged.index=pd.to_datetime(merged['time'])
merged.drop(['time'],inplace=True,axis=1)

co=merged.corr()
stat['Correlation'].loc[4]=co['WTMP'][8]
stat['Bias'].loc[4]=np.mean(merged['WTMP']-merged[8])
stat['RMSE'].loc[4]=np.sqrt(np.sum((merged['WTMP']-merged[8])**2)/len(merged))
stat['Index of Agreement'].loc[4]=1-np.sum((merged['WTMP']-merged[8])**2)/np.sum((merged['WTMP']-merged['WTMP'].mean())**2+(merged[8]-merged[8].mean())**2)
stat['STD'].loc[4]=np.std(merged[8])

# %% 44099

dfm=pd.DataFrame(df_model1[9])
dfm=dfm.loc[datetime(2018,2,1):datetime(2019,2,1)]
dfm['time']=dfm.index

dfo=pd.DataFrame(pd.to_numeric(df_44099['WTMP']))
dfo.index=pd.to_datetime(dfo.index)
dfo=dfo.loc[datetime(2018,2,1):datetime(2019,2,1)]
dfo=dfo['WTMP'].loc[dfo['WTMP'] != 999.0 ]
dfo=pd.DataFrame(dfo)
dfo=dfo.resample('30T').ffill(limit=1).interpolate(method='linear',axis=1) # you cannot increase res of data with this method
dfo.dropna(inplace=True)
dfo['time']=dfo.index

merged=pd.merge(dfo, dfm, on='time',how='outer')
merged.dropna(inplace=True)
merged.index=pd.to_datetime(merged['time'])
merged.drop(['time'],inplace=True,axis=1)

co=merged.corr()
stat['Correlation'].loc[5]=co['WTMP'][9]
stat['Bias'].loc[5]=np.mean(merged['WTMP']-merged[9])
stat['RMSE'].loc[5]=np.sqrt(np.sum((merged['WTMP']-merged[9])**2)/len(merged))
stat['Index of Agreement'].loc[5]=1-np.sum((merged['WTMP']-merged[9])**2)/np.sum((merged['WTMP']-merged['WTMP'].mean())**2+(merged[9]-merged[9].mean())**2)
stat['STD'].loc[5]=np.std(merged[9])

# %% 44089

dfm=pd.DataFrame(df_model1[10])
dfm=dfm.loc[datetime(2018,2,1):datetime(2019,2,1)]
dfm['time']=dfm.index

dfo=pd.DataFrame(pd.to_numeric(df_44089['WTMP']))
dfo.index=pd.to_datetime(dfo.index)
dfo=dfo.loc[datetime(2018,2,1):datetime(2019,2,1)]
dfo=dfo['WTMP'].loc[dfo['WTMP'] != 999.0 ]
dfo=pd.DataFrame(dfo)
dfo=dfo.resample('30T').ffill(limit=1).interpolate(method='linear',axis=1) # you cannot increase res of data with this method
dfo.dropna(inplace=True)
dfo['time']=dfo.index

merged=pd.merge(dfo, dfm, on='time',how='outer')
merged.dropna(inplace=True)
merged.index=pd.to_datetime(merged['time'])
merged.drop(['time'],inplace=True,axis=1)

co=merged.corr()
stat['Correlation'].loc[6]=co['WTMP'][10]
stat['Bias'].loc[6]=np.mean(merged['WTMP']-merged[10])
stat['RMSE'].loc[6]=np.sqrt(np.sum((merged['WTMP']-merged[10])**2)/len(merged))
stat['Index of Agreement'].loc[6]=1-np.sum((merged['WTMP']-merged[10])**2)/np.sum((merged['WTMP']-merged['WTMP'].mean())**2+(merged[10]-merged[10].mean())**2)
stat['STD'].loc[6]=np.std(merged[10])

# %% 44009

dfm=pd.DataFrame(df_model1[11])
#dfm.rename(columns={11:'temp'},inplace=True)
dfm=dfm.loc[datetime(2018,2,1):datetime(2019,2,1)]
dfm['time']=dfm.index

dfo=pd.DataFrame(pd.to_numeric(df_44009['WTMP']))
#dfo.rename(columns={'WTMP':'temp'},inplace=True)
dfo=dfo.loc[datetime(2018,2,1):datetime(2019,2,1)]
dfo=dfo['WTMP'].loc[dfo['WTMP'] != 999.0 ]
dfo=pd.DataFrame(dfo)
dfo['time']=dfo.index

merged=pd.merge(dfo, dfm, on='time',how='outer')
merged.dropna(inplace=True)
merged.index=pd.to_datetime(merged['time'])
merged.drop(['time'],inplace=True,axis=1)

co=merged.corr()
stat['Correlation'].loc[7]=co['WTMP'][11]
stat['Bias'].loc[7]=np.mean(merged['WTMP']-merged[11])
stat['RMSE'].loc[7]=np.sqrt(np.sum((merged['WTMP']-merged[11])**2)/len(merged))
stat['Index of Agreement'].loc[7]=1-np.sum((merged['WTMP']-merged[11])**2)/np.sum((merged['WTMP']-merged['WTMP'].mean())**2+(merged[11]-merged[11].mean())**2)
stat['STD'].loc[7]=np.std(merged[11])

# %% 44091

dfm=pd.DataFrame(df_model1[12])
dfm=dfm.loc[datetime(2018,2,1):datetime(2019,2,1)]
dfm['time']=dfm.index

dfo=pd.DataFrame(pd.to_numeric(df_44091['WTMP']))
dfo.index=pd.to_datetime(dfo.index)
dfo=dfo.loc[datetime(2018,2,1):datetime(2019,2,1)]
dfo=dfo['WTMP'].loc[dfo['WTMP'] != 999.0 ]
dfo=pd.DataFrame(dfo)
dfo=dfo.resample('30T').ffill(limit=1).interpolate(method='linear',axis=1) # you cannot increase res of data with this method
dfo.dropna(inplace=True)
dfo['time']=dfo.index

merged=pd.merge(dfo, dfm, on='time',how='outer')
merged.dropna(inplace=True)
merged.index=pd.to_datetime(merged['time'])
merged.drop(['time'],inplace=True,axis=1)

co=merged.corr()
stat['Correlation'].loc[8]=co['WTMP'][12]
stat['Bias'].loc[8]=np.mean(merged['WTMP']-merged[12])
stat['RMSE'].loc[8]=np.sqrt(np.sum((merged['WTMP']-merged[12])**2)/len(merged))
stat['Index of Agreement'].loc[8]=1-np.sum((merged['WTMP']-merged[12])**2)/np.sum((merged['WTMP']-merged['WTMP'].mean())**2+(merged[12]-merged[12].mean())**2)
stat['STD'].loc[8]=np.std(merged[12])

# %% 44065

dfm=pd.DataFrame(df_model1[13])
dfm=dfm.loc[datetime(2018,2,1):datetime(2019,2,1)]
dfm['time']=dfm.index

dfo=pd.DataFrame(pd.to_numeric(df_44065['WTMP']))
dfo.index=pd.to_datetime(dfo.index)
dfo=dfo.loc[datetime(2018,2,1):datetime(2019,2,1)]
dfo=dfo['WTMP'].loc[dfo['WTMP'] != 999.0 ]
dfo=pd.DataFrame(dfo)
dfo=dfo.resample('30T').ffill(limit=1).interpolate(method='linear',axis=1) # you cannot increase res of data with this method
dfo.dropna(inplace=True)
dfo['time']=dfo.index

merged=pd.merge(dfo, dfm, on='time',how='outer')
merged.dropna(inplace=True)
merged.index=pd.to_datetime(merged['time'])
merged.drop(['time'],inplace=True,axis=1)

co=merged.corr()
stat['Correlation'].loc[9]=co['WTMP'][13]
stat['Bias'].loc[9]=np.mean(merged['WTMP']-merged[13])
stat['RMSE'].loc[9]=np.sqrt(np.sum((merged['WTMP']-merged[13])**2)/len(merged))
stat['Index of Agreement'].loc[9]=1-np.sum((merged['WTMP']-merged[13])**2)/np.sum((merged['WTMP']-merged['WTMP'].mean())**2+(merged[13]-merged[13].mean())**2)
stat['STD'].loc[9]=np.std(merged[13])

# %% 44025

dfm=pd.DataFrame(df_model1[14])
dfm=dfm.loc[datetime(2018,2,1):datetime(2019,2,1)]
dfm['time']=dfm.index

dfo=pd.DataFrame(pd.to_numeric(df_44025['WTMP']))
dfo.index=pd.to_datetime(dfo.index)
dfo=dfo.loc[datetime(2018,2,1):datetime(2019,2,1)]
dfo=dfo['WTMP'].loc[dfo['WTMP'] != 999.0 ]
dfo=pd.DataFrame(dfo)
dfo=dfo.resample('30T').ffill(limit=1).interpolate(method='linear',axis=1) # you cannot increase res of data with this method
dfo.dropna(inplace=True)
dfo['time']=dfo.index

merged=pd.merge(dfo, dfm, on='time',how='outer')
merged.dropna(inplace=True)
merged.index=pd.to_datetime(merged['time'])
merged.drop(['time'],inplace=True,axis=1)

co=merged.corr()
stat['Correlation'].loc[10]=co['WTMP'][14]
stat['Bias'].loc[10]=np.mean(merged['WTMP']-merged[14])
stat['RMSE'].loc[10]=np.sqrt(np.sum((merged['WTMP']-merged[14])**2)/len(merged))
stat['Index of Agreement'].loc[10]=1-np.sum((merged['WTMP']-merged[14])**2)/np.sum((merged['WTMP']-merged['WTMP'].mean())**2+(merged[14]-merged[14].mean())**2)
stat['STD'].loc[10]=np.std(merged[14])

# %% 44066

dfm=pd.DataFrame(df_model1[15])
dfm=dfm.loc[datetime(2018,2,1):datetime(2019,2,1)]
dfm['time']=dfm.index

dfo=pd.DataFrame(pd.to_numeric(df_44066['WTMP']))
dfo.index=pd.to_datetime(dfo.index)
dfo=dfo.loc[datetime(2018,2,1):datetime(2019,2,1)]
dfo=dfo['WTMP'].loc[dfo['WTMP'] != 999.0 ]
dfo=pd.DataFrame(dfo)
dfo=dfo.resample('30T').ffill(limit=1).interpolate(method='linear',axis=1) # you cannot increase res of data with this method
dfo.dropna(inplace=True)
dfo['time']=dfo.index

merged=pd.merge(dfo, dfm, on='time',how='outer')
merged.dropna(inplace=True)
merged.index=pd.to_datetime(merged['time'])
merged.drop(['time'],inplace=True,axis=1)

co=merged.corr()
stat['Correlation'].loc[11]=co['WTMP'][15]
stat['Bias'].loc[11]=np.mean(merged['WTMP']-merged[15])
stat['RMSE'].loc[11]=np.sqrt(np.sum((merged['WTMP']-merged[15])**2)/len(merged))
stat['Index of Agreement'].loc[11]=1-np.sum((merged['WTMP']-merged[15])**2)/np.sum((merged['WTMP']-merged['WTMP'].mean())**2+(merged[15]-merged[15].mean())**2)
stat['STD'].loc[11]=np.std(merged[15])

# %% 

print(stat)
#stat=pd.to_numeric(stat)



# %% Taylor Diagram

#
#for index,row in stat.iterrows():
STD=stat['STD'].values.astype(float)
RMSE=stat['RMSE'].values.astype(float)
Corr=stat['Correlation'].values.astype(float)
label=list(stat['St_name'])

for i in range(len(label)):
    label[i]=label[i][0:25]

fig=plt.figure(figsize=(10,10))
ax=taylor_diagram(STD,RMSE,Corr,markerLabel=label,
                          markerLegend ='on', markerColor = 'r',
                          styleOBS = '-', colOBS = 'r', markerobs = 'o',
                          markerSize = 10, tickRMS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                          tickRMSangle = 115, showlabelsRMS = 'on',
                          titleRMS = 'on', titleOBS = 'Ref')

#ax=taylor_diagram(STD,RMSE,Corr,markerLabel=label,
#                          markerLegend ='on', markerColor = 'r',
#                          styleOBS = '-', colOBS = 'r', markerobs = 'o',
#                          markerSize = 10, tickRMS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
#                          tickRMSangle = 115, showlabelsRMS = 'on',
#                          titleRMS = 'on', titleOBS = 'Ref')
#ax.set_title("NDBC Stations")

plt.savefig('taylor_WL_CU_sd4.png')
plt.show()

# %% 
stat['St_name']=label
stat.to_csv('stats_ndbc_wl_CU_sd4.csv')

# %% 



# %% 