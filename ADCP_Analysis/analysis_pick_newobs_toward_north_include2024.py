### Created by Amin Ilia                      ###
### program to anlysis ADCP data plot results ###
### Directions are toward north               ###

# %% import libraries 
import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation 
from matplotlib.dates import DateFormatter
from PIL import Image
from datetime import datetime
from windrose import WindroseAxes
import matplotlib.cm as cm

# %% Read ADCP data file 

os.chdir(r'C:\work\CSA\ADCP\new_obs')

df2=pd.read_csv('Currents ADCP 3_HA3102_reformat.csv', header=[0,1], index_col=[0])
df3=pd.read_csv('Currents ADCP 2_HA2103_reformat.csv', header=[0,1], index_col=[0])
df4=pd.read_csv('Currents ADCP1_ HA1104_reformat.csv', header=[0,1], index_col=[0])

df2.index=pd.to_datetime(df2.index)
df3.index=pd.to_datetime(df3.index)
df4.index=pd.to_datetime(df4.index)

# %% 

dp2=np.array(df2['Pressure'].mean())-np.arange(2,14)
dp3=np.array(df3['Pressure'].mean())-np.arange(2,10)
dp4=np.array(df4['Pressure'].mean())-np.arange(2,8)


# %% 
#dp2=np.arange(2,14)
#dp3=np.arange(2,10)
#dp4=np.arange(2,8)

# %% 

df2=df2.dropna()
df3=df3.dropna()
df4=df4.dropna()


df2.replace(-9.0,np.nan, inplace=True)
df3.replace(-9.0,np.nan, inplace=True)

dfeM=df2['Mag']
dfeD=df2['Dir']

dfmM=df3['Mag']
dfmD=df3['Dir']

dfwM=df4['Mag']
dfwD=df4['Dir']

# %% 

print(df2['Mag'].max().max())
print(df2['Mag'].min().min())

print(df3['Mag'].max().max())
print(df3['Mag'].min().min())

print(df4['Mag'].max().max())
print(df4['Mag'].min().min())


df2=df2.dropna()
df3=df3.dropna()
df4=df4.dropna()

# %% 

dfeu=dfeM*np.cos((90-dfeD)/180*np.pi)
dfev=dfeM*np.sin((90-dfeD)/180*np.pi)

dfmu=dfmM*np.cos((90-dfmD)/180*np.pi)
dfmv=dfmM*np.sin((90-dfmD)/180*np.pi)

dfwu=dfwM*np.cos((90-dfwD)/180*np.pi)
dfwv=dfwM*np.sin((90-dfwD)/180*np.pi)

# %% estimate mean and sigma f

dfeum=dfeu.mean(axis=1)
dfevm=dfev.mean(axis=1)
ustde=np.std(dfeum)
ume=np.mean(dfeum)
vstde=np.std(dfevm)
vme=np.mean(dfevm)

dfmum=dfmu.mean(axis=1)
dfmvm=dfmv.mean(axis=1)
ustdm=np.std(dfmum)
umm=np.mean(dfmum)
vstdm=np.std(dfmvm)
vmm=np.mean(dfmvm)

dfwum=dfwu.mean(axis=1)
dfwvm=dfwv.mean(axis=1)
ustdw=np.std(dfwum)
umw=np.mean(dfwum)
vstdw=np.std(dfwvm)
vmw=np.mean(dfwvm)

# %% extract upper and lower bounds 

dfeup=dfeu.loc[dfeum[(dfeum) > (ume+ustde)].index]
dfeun=dfeu.loc[dfeum[(dfeum) < (ume-ustde)].index]
dfevp=dfev.loc[dfevm[(dfevm) > (vme+vstde)].index]
dfevn=dfev.loc[dfevm[(dfevm) < (vme-vstde)].index]

dfmup=dfmu.loc[dfmum[(dfmum) > (umm+ustdm)].index]
dfmun=dfmu.loc[dfmum[(dfmum) < (umm-ustdm)].index]
dfmvp=dfmv.loc[dfmvm[(dfmvm) > (vmm+vstdm)].index]
dfmvn=dfmv.loc[dfmvm[(dfmvm) < (vmm-vstdm)].index]

dfwup=dfwu.loc[dfwum[(dfwum) > (umw+ustdw)].index]
dfwun=dfwu.loc[dfwum[(dfwum) < (umw-ustdw)].index]
dfwvp=dfwv.loc[dfwvm[(dfwvm) > (vmw+vstdw)].index]
dfwvn=dfwv.loc[dfwvm[(dfwvm) < (vmw-vstdw)].index]

# %% Ploting mean profiles of lower and higher bounds

fig, ax=plt.subplots(1,2,figsize=(9,5))
ax[0].plot(dfeup.mean(),dp2,'b',linewidth=3)
ax[0].plot(dfeun.mean(),dp2,'r',linewidth=3)
ax[0].legend(['Upper Bound','lower Bound'],fontsize=10)
ax[0].set_title('U - Upper/Lower Bounds - Southwest Field')
ax[0].set(xlim=[-0.7,0.7])
ax[0].set(ylim=[0.00,14.0])
ax[0].set_xlabel('Speed (m/s)', size=10)
ax[0].set_ylabel('Depth (m)', size=10)
ax[0].invert_yaxis()
ax[0].grid(linewidth=0.3)

ax[1].plot(dfevp.mean(),dp2,'b',linewidth=3)
ax[1].plot(dfevn.mean(),dp2,'r',linewidth=3)
ax[1].legend(['Upper Bound','lower Bound'],fontsize=10)
ax[1].set_title('V - Upper/Lower Bounds - Southwest Field')
ax[1].set(xlim=[-1.2,1.2])
ax[1].set(ylim=[0.00,14.0])
ax[1].set_xlabel('Speed (m/s)', size=10)
ax[1].set_ylabel('Depth (m)', size=10)
ax[1].invert_yaxis()
ax[1].grid(linewidth=0.3)
plt.savefig('HA3102_General_Profiles.png',dpi=300)


fig, ax=plt.subplots(1,2,figsize=(9,5))
ax[0].plot(dfmup.mean(),dp3,'b',linewidth=3)
ax[0].plot(dfmun.mean(),dp3,'r',linewidth=3)
ax[0].legend(['Upper Bound','lower Bound'],fontsize=10)
ax[0].set_title('U - Upper/Lower Bound - Main Field')
ax[0].set(xlim=[-0.6,0.6])
ax[0].set(ylim=[0.00,12.0])
ax[0].set_xlabel('Speed (m/s)', size=10)
ax[0].set_ylabel('Depth (m)', size=10)
ax[0].invert_yaxis()
ax[0].grid(linewidth=0.3)

ax[1].plot(dfmvp.mean(),dp3,'b',linewidth=3)
ax[1].plot(dfmvn.mean(),dp3,'r',linewidth=3)
ax[1].legend(['Upper Bound','lower Bound'],fontsize=10)
ax[1].set_title('V - Upper/Lower Bounds - Main Field')
ax[1].set(xlim=[-0.6,0.6])
ax[1].set(ylim=[0.00,10.0])
ax[1].set_xlabel('Speed (m/s)', size=10)
ax[1].set_ylabel('Depth (m)', size=10)
ax[1].invert_yaxis()
ax[1].grid(linewidth=0.3)
plt.savefig('HA2103_General_Profiles.png',dpi=300)


fig, ax=plt.subplots(1,2,figsize=(9,5))
ax[0].plot(dfwup.mean(),dp4,'b',linewidth=3)
ax[0].plot(dfwun.mean(),dp4,'r',linewidth=3)
ax[0].legend(['Upper Bound','lower Bound'],fontsize=10)
ax[0].set_title('U - Upper/Lower Bounds - East Field')
ax[0].set(xlim=[-0.4,0.4])
ax[0].set(ylim=[0.00,8.0])
ax[0].set_xlabel('Speed (m/s)', size=10)
ax[0].set_ylabel('Depth (m)', size=10)
ax[0].invert_yaxis()
ax[0].grid(linewidth=0.3)

ax[1].plot(dfwvp.mean(),dp4,'b',linewidth=3)
ax[1].plot(dfwvn.mean(),dp4,'r',linewidth=3)
ax[1].legend(['Upper Bound','lower Bound'],fontsize=10)
ax[1].set_title('V - Upper/Lower Bounds - East Field')
ax[1].set(xlim=[-0.1,0.1])
ax[1].set(ylim=[0.00,8.0])
ax[1].set_xlabel('Speed (m/s)', size=10)
ax[1].set_ylabel('Depth (m)', size=10)
ax[1].invert_yaxis()
ax[1].grid(linewidth=0.3)
plt.savefig('HA1104_General_Profiles.png',dpi=300)

# %% plot current rose

fig, ax=plt.subplots(subplot_kw={'projection':'windrose'})
ax.bar(dfeD['12'],dfeM['12'], normed=True,edgecolor='white' ,cmap=cm.cool)
ax.set_legend()
plt.suptitle('HA3102 Upper Layer Current Rose')
plt.savefig('HA3102_upper_Layer_Rose_.png')

fig, ax=plt.subplots(subplot_kw={'projection':'windrose'})
ax.bar(dfeD['6'],dfeM['6'], normed=True,edgecolor='white' ,cmap=cm.cool)
ax.set_legend()
plt.suptitle('HA3102 Middle Layer Current Rose')
plt.savefig('HA3102_Middle_Layer_Rose_.png')

fig, ax=plt.subplots(subplot_kw={'projection':'windrose'})
ax.bar(dfeD['1'],dfeM['1'], normed=True,edgecolor='white' ,cmap=cm.cool)
ax.set_legend()
plt.suptitle('HA3102 Bottom Layer Current Rose')
plt.savefig('HA3102_Bottom_Layer_Rose_.png')


fig, ax=plt.subplots(subplot_kw={'projection':'windrose'})
ax.bar(dfmD['8'],dfmM['8'], normed=True,edgecolor='white' ,cmap=cm.cool)
ax.set_legend()
plt.suptitle('HA2103 Upper Layer Current Rose')
plt.savefig('HA2103_Upper_Layer_Rose_.png')

fig, ax=plt.subplots(subplot_kw={'projection':'windrose'})
ax.bar(dfmD['4'],dfmM['4'], normed=True,edgecolor='white' ,cmap=cm.cool)
ax.set_legend()
plt.suptitle('HA2103 Middle Layer Current Rose')
plt.savefig('HA2103_Middle_Layer_Rose_.png')

fig, ax=plt.subplots(subplot_kw={'projection':'windrose'})
ax.bar(dfmD['1'],dfmM['1'], normed=True,edgecolor='white' ,cmap=cm.cool)
ax.set_legend()
plt.suptitle('HA2103 Bottom Layer Current Rose')
plt.savefig('HA2103_Bottom_Layer_Rose_.png')


fig, ax=plt.subplots(subplot_kw={'projection':'windrose'})
ax.bar(dfwD['6'],dfwM['6'], normed=True,edgecolor='white' ,cmap=cm.cool)
ax.set_legend()
plt.suptitle('HA1104 Field Upper Layer Current Rose')
plt.savefig('HA1104_Upper_Layer_Rose_.png')

fig, ax=plt.subplots(subplot_kw={'projection':'windrose'})
ax.bar(dfwD['3'],dfwM['3'], normed=True,edgecolor='white' ,cmap=cm.cool)
ax.set_legend()
plt.suptitle('HA1104 Field Middle Layer Current Rose')
plt.savefig('HA1104_Middle_Layer_Rose_.png')

fig, ax=plt.subplots(subplot_kw={'projection':'windrose'})
ax.bar(dfwD['1'],dfwM['1'], normed=True,edgecolor='white' ,cmap=cm.cool)
ax.set_legend()
plt.suptitle('HA1104 Bottom Layer Current Rose')
plt.savefig('HA1104_Bottom_Layer_Rose_.png')

# %% 

###################################################
###################################################
###################################################
############## New set of ADCP 2014 ###############
###################################################
###################################################
###################################################

# %% Read ADCP data file 

os.chdir(r'C:\work\CSA\ADCP\new_obs')

df2=pd.read_csv('Currents ADCP 3 HA3204_Dec 2023 to Jan 2024.csv', header=[0,1], index_col=[0])
df4=pd.read_csv('Currents ADCP 1 HA1205_Dec 2023 to Jan 2024.csv', header=[0,1], index_col=[0])

df2.index=pd.to_datetime(df2.index)
df4.index=pd.to_datetime(df4.index)

# %% 

dp2=np.array(df2['Pressure'].mean())-np.arange(2,14)
dp4=np.array(df4['Pressure'].mean())-np.arange(2,8)


# %% 
#dp2=np.arange(2,14)
#dp3=np.arange(2,10)
#dp4=np.arange(2,8)

# %% 

df2=df2.dropna()
df4=df4.dropna()

df2.replace(-9.0,np.nan, inplace=True)
df4.replace(-9.0,np.nan, inplace=True)

dfeM=df2['Mag']
dfeD=df2['Dir']

dfwM=df4['Mag']
dfwD=df4['Dir']

# %% 

print(df2['Mag'].max().max())
print(df2['Mag'].min().min())

print(df4['Mag'].max().max())
print(df4['Mag'].min().min())

df2=df2.dropna()
df4=df4.dropna()

# %% 

dfeu=dfeM*np.cos((90-dfeD)/180*np.pi)
dfev=dfeM*np.sin((90-dfeD)/180*np.pi)

dfwu=dfwM*np.cos((90-dfwD)/180*np.pi)
dfwv=dfwM*np.sin((90-dfwD)/180*np.pi)

# %% estimate mean and sigma f

dfeum=dfeu.mean(axis=1)
dfevm=dfev.mean(axis=1)
ustde=np.std(dfeum)
ume=np.mean(dfeum)
vstde=np.std(dfevm)
vme=np.mean(dfevm)

dfwum=dfwu.mean(axis=1)
dfwvm=dfwv.mean(axis=1)
ustdw=np.std(dfwum)
umw=np.mean(dfwum)
vstdw=np.std(dfwvm)
vmw=np.mean(dfwvm)

# %% extract upper and lower bounds 

dfeup=dfeu.loc[dfeum[(dfeum) > (ume+ustde)].index]
dfeun=dfeu.loc[dfeum[(dfeum) < (ume-ustde)].index]
dfevp=dfev.loc[dfevm[(dfevm) > (vme+vstde)].index]
dfevn=dfev.loc[dfevm[(dfevm) < (vme-vstde)].index]

dfwup=dfwu.loc[dfwum[(dfwum) > (umw+ustdw)].index]
dfwun=dfwu.loc[dfwum[(dfwum) < (umw-ustdw)].index]
dfwvp=dfwv.loc[dfwvm[(dfwvm) > (vmw+vstdw)].index]
dfwvn=dfwv.loc[dfwvm[(dfwvm) < (vmw-vstdw)].index]

# %% Ploting mean profiles of lower and higher bounds

fig, ax=plt.subplots(1,2,figsize=(9,5))
ax[0].plot(dfeup.mean(),dp2,'b',linewidth=3)
ax[0].plot(dfeun.mean(),dp2,'r',linewidth=3)
ax[0].legend(['Upper Bound','lower Bound'],fontsize=10)
ax[0].set_title('U - Upper/Lower Bounds - Southwest Field')
ax[0].set(xlim=[-0.7,0.7])
ax[0].set(ylim=[0.00,14.0])
ax[0].set_xlabel('Speed (m/s)', size=10)
ax[0].set_ylabel('Depth (m)', size=10)
ax[0].invert_yaxis()
ax[0].grid(linewidth=0.3)

ax[1].plot(dfevp.mean(),dp2,'b',linewidth=3)
ax[1].plot(dfevn.mean(),dp2,'r',linewidth=3)
ax[1].legend(['Upper Bound','lower Bound'],fontsize=10)
ax[1].set_title('V - Upper/Lower Bounds - Southwest Field')
ax[1].set(xlim=[-1.2,1.2])
ax[1].set(ylim=[0.00,14.0])
ax[1].set_xlabel('Speed (m/s)', size=10)
ax[1].set_ylabel('Depth (m)', size=10)
ax[1].invert_yaxis()
ax[1].grid(linewidth=0.3)
plt.savefig('HA3102_General_Profiles_2024.png',dpi=300)

# %% 

fig, ax=plt.subplots(1,2,figsize=(9,5))
ax[0].plot(dfwup.mean(),dp4,'b',linewidth=3)
ax[0].plot(dfwun.mean(),dp4,'r',linewidth=3)
ax[0].legend(['Upper Bound','lower Bound'],fontsize=10)
ax[0].set_title('U - Upper/Lower Bounds - East Field')
ax[0].set(xlim=[-0.4,0.4])
ax[0].set(ylim=[0.00,8.0])
ax[0].set_xlabel('Speed (m/s)', size=10)
ax[0].set_ylabel('Depth (m)', size=10)
ax[0].invert_yaxis()
ax[0].grid(linewidth=0.3)

ax[1].plot(dfwvp.mean(),dp4,'b',linewidth=3)
ax[1].plot(dfwvn.mean(),dp4,'r',linewidth=3)
ax[1].legend(['Upper Bound','lower Bound'],fontsize=10)
ax[1].set_title('V - Upper/Lower Bounds - East Field')
ax[1].set(xlim=[-0.1,0.1])
ax[1].set(ylim=[0.00,8.0])
ax[1].set_xlabel('Speed (m/s)', size=10)
ax[1].set_ylabel('Depth (m)', size=10)
ax[1].invert_yaxis()
ax[1].grid(linewidth=0.3)
plt.savefig('HA1104_General_Profiles._2024.png',dpi=300)

# %% plot current rose

fig, ax=plt.subplots(subplot_kw={'projection':'windrose'})
ax.bar(dfeD['12'],dfeM['12'], normed=True,edgecolor='white' ,cmap=cm.cool)
ax.set_legend()
plt.suptitle('HA3102 Upper Layer Current Rose')
plt.savefig('HA3102_upper_Layer_Rose_2014.png')

fig, ax=plt.subplots(subplot_kw={'projection':'windrose'})
ax.bar(dfeD['6'],dfeM['6'], normed=True,edgecolor='white' ,cmap=cm.cool)
ax.set_legend()
plt.suptitle('HA3102 Middle Layer Current Rose')
plt.savefig('HA3102_Middle_Layer_Rose_2014.png')

fig, ax=plt.subplots(subplot_kw={'projection':'windrose'})
ax.bar(dfeD['1'],dfeM['1'], normed=True,edgecolor='white' ,cmap=cm.cool)
ax.set_legend()
plt.suptitle('HA3102 Bottom Layer Current Rose')
plt.savefig('HA3102_Bottom_Layer_Rose_2014.png')



fig, ax=plt.subplots(subplot_kw={'projection':'windrose'})
ax.bar(dfwD['6'],dfwM['6'], normed=True,edgecolor='white' ,cmap=cm.cool)
ax.set_legend()
plt.suptitle('HA1104 Field Upper Layer Current Rose')
plt.savefig('HA1104_Upper_Layer_Rose_2014.png')

fig, ax=plt.subplots(subplot_kw={'projection':'windrose'})
ax.bar(dfwD['3'],dfwM['3'], normed=True,edgecolor='white' ,cmap=cm.cool)
ax.set_legend()
plt.suptitle('HA1104 Field Middle Layer Current Rose')
plt.savefig('HA1104_Middle_Layer_Rose_2014.png')

fig, ax=plt.subplots(subplot_kw={'projection':'windrose'})
ax.bar(dfwD['1'],dfwM['1'], normed=True,edgecolor='white' ,cmap=cm.cool)
ax.set_legend()
plt.suptitle('HA1104 Bottom Layer Current Rose')
plt.savefig('HA1104_Bottom_Layer_Rose_2014.png')

# %% 

