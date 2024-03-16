### Created by Amin Ilia                      ###
### program to anlysis ADCP data plot results ###

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

df2=pd.read_csv('Currents ADCP 3_HA3102.csv', header=[0], index_col=[0])
df3=pd.read_csv('Currents ADCP 2_HA2103.csv', header=[0], index_col=[0])
df4=pd.read_csv('Currents ADCP1_ HA1104.csv', header=[0], index_col=[0])

df2.index=pd.to_datetime(df2.index)
df3.index=pd.to_datetime(df3.index)
df4.index=pd.to_datetime(df4.index)

df2 = df2.reindex(sorted(df2.columns), axis=1)
df3 = df3.reindex(sorted(df3.columns), axis=1)
df4 = df4.reindex(sorted(df4.columns), axis=1)

df2.to_csv('Currents ADCP 3_HA3102_reformat.csv')
df3.to_csv('Currents ADCP 2_HA2103_reformat.csv')
df4.to_csv('Currents ADCP1_ HA1104_reformat.csv')

# %%

df2=df2.dropna()
df3=df3.dropna()
df4=df4.dropna()

# %% 

df2=df2['Mag']
dfeD=dfe['Dir']
dfeu=dfe['Eas']
dfev=dfe['Nor']

dfmM=dfm['Mag']
dfmD=dfm['Dir']
dfmu=dfm['Eas']
dfmv=dfm['Nor']

dfwM=dfw['Mag']
dfwD=dfw['Dir']
dfwu=dfw['Eas']
dfwv=dfw['Nor']

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
ax[0].plot(dfeup.mean(),dp2['Depth Below Surface'],'b',linewidth=3)
ax[0].plot(dfeun.mean(),dp2['Depth Below Surface'],'r',linewidth=3)
ax[0].legend(['East','West'],fontsize=10)
ax[0].set_title('U - Upper/Lower Bounds - East Field')
ax[0].set(xlim=[-0.35,0.35])
ax[0].set(ylim=[0.00,12.0])
ax[0].set_xlabel('Speed (m/s)', size=10)
ax[0].set_ylabel('Depth (m)', size=10)
ax[0].invert_yaxis()
ax[0].grid(linewidth=0.3)

ax[1].plot(dfevp.mean(),dp2['Depth Below Surface'],'b',linewidth=3)
ax[1].plot(dfevn.mean(),dp2['Depth Below Surface'],'r',linewidth=3)
ax[1].legend(['East','West'],fontsize=10)
ax[1].set_title('V - Upper/Lower Bounds - East Field')
ax[1].set(xlim=[-0.35,0.35])
ax[1].set(ylim=[0.00,12.0])
ax[1].set_xlabel('Speed (m/s)', size=10)
ax[1].set_ylabel('Depth (m)', size=10)
ax[1].invert_yaxis()
ax[1].grid(linewidth=0.3)
plt.savefig('East_General_Profiles_.png',dpi=300)


fig, ax=plt.subplots(1,2,figsize=(9,5))
ax[0].plot(dfmup.mean(),dp3['Depth Below Surface'],'b',linewidth=3)
ax[0].plot(dfmun.mean(),dp3['Depth Below Surface'],'r',linewidth=3)
ax[0].legend(['East','West'],fontsize=10)
ax[0].set_title('U - Upper/Lower Bound - Main Field')
ax[0].set(xlim=[-0.4,0.4])
ax[0].set(ylim=[0.00,12.0])
ax[0].set_xlabel('Speed (m/s)', size=10)
ax[0].set_ylabel('Depth (m)', size=10)
ax[0].invert_yaxis()
ax[0].grid(linewidth=0.3)

ax[1].plot(dfmvp.mean(),dp3['Depth Below Surface'],'b',linewidth=3)
ax[1].plot(dfmvn.mean(),dp3['Depth Below Surface'],'r',linewidth=3)
ax[1].legend(['East','West'],fontsize=10)
ax[1].set_title('V - Upper/Lower Bounds - Main Field')
ax[1].set(xlim=[-0.4,0.4])
ax[1].set(ylim=[0.00,12.0])
ax[1].set_xlabel('Speed (m/s)', size=10)
ax[1].set_ylabel('Depth (m)', size=10)
ax[1].invert_yaxis()
ax[1].grid(linewidth=0.3)
plt.savefig('Main_General_Profiles_.png',dpi=300)


fig, ax=plt.subplots(1,2,figsize=(9,5))
ax[0].plot(dfwup.mean(),dp4['Depth Below Surface'],'b',linewidth=3)
ax[0].plot(dfwun.mean(),dp4['Depth Below Surface'],'r',linewidth=3)
ax[0].legend(['East','West'],fontsize=10)
ax[0].set_title('U - Upper/Lower Bounds - Southwest Field')
ax[0].set(xlim=[-0.55,0.55])
ax[0].set(ylim=[0.00,17.0])
ax[0].set_xlabel('Speed (m/s)', size=10)
ax[0].set_ylabel('Depth (m)', size=10)
ax[0].invert_yaxis()
ax[0].grid(linewidth=0.3)

ax[1].plot(dfwvp.mean(),dp4['Depth Below Surface'],'b',linewidth=3)
ax[1].plot(dfwvn.mean(),dp4['Depth Below Surface'],'r',linewidth=3)
ax[1].legend(['East','West'],fontsize=10)
ax[1].set_title('V - Upper/Lower Bounds - Southwest Field')
ax[1].set(xlim=[-1.0,1.0])
ax[1].set(ylim=[0.00,17.0])
ax[1].set_xlabel('Speed (m/s)', size=10)
ax[1].set_ylabel('Depth (m)', size=10)
ax[1].invert_yaxis()
ax[1].grid(linewidth=0.3)
plt.savefig('Southwest_General_Profiles_.png',dpi=300)


# %% plot current rose

fig, ax=plt.subplots(subplot_kw={'projection':'windrose'})
ax.bar(dfeD['1'],dfeM['1'], normed=True,edgecolor='white' ,cmap=cm.cool)
ax.set_legend()
plt.suptitle('East Field Bottom Layer Current Rose')
plt.savefig('Eastfields_Bottom_Layer_Rose_.png')

fig, ax=plt.subplots(subplot_kw={'projection':'windrose'})
ax.bar(dfeD['6'],dfeM['6'], normed=True,edgecolor='white' ,cmap=cm.cool)
ax.set_legend()
plt.suptitle('East Field Middle Layer Current Rose')
plt.savefig('Eastfields_Middle_Layer_Rose_.png')

fig, ax=plt.subplots(subplot_kw={'projection':'windrose'})
ax.bar(dfeD['10'],dfeM['10'], normed=True,edgecolor='white' ,cmap=cm.cool)
ax.set_legend()
plt.suptitle('East Field Upper Layer Current Rose')
plt.savefig('Eastfiels_Upper_Layer_Rose_.png')


fig, ax=plt.subplots(subplot_kw={'projection':'windrose'})
ax.bar(dfmD['1'],dfmM['1'], normed=True,edgecolor='white' ,cmap=cm.cool)
ax.set_legend()
plt.suptitle('Main Field Bottom Layer Current Rose')
plt.savefig('Mainfields_Bottom_Layer_Rose_.png')

fig, ax=plt.subplots(subplot_kw={'projection':'windrose'})
ax.bar(dfmD['6'],dfmM['6'], normed=True,edgecolor='white' ,cmap=cm.cool)
ax.set_legend()
plt.suptitle('Main Field Middle Layer Current Rose')
plt.savefig('Mainfields_Middle_Layer_Rose_.png')

fig, ax=plt.subplots(subplot_kw={'projection':'windrose'})
ax.bar(dfmD['11'],dfmM['11'], normed=True,edgecolor='white' ,cmap=cm.cool)
ax.set_legend()
plt.suptitle('Main Field Upper Layer Current Rose')
plt.savefig('Mainfiels_Upper_Layer_Rose_.png')


fig, ax=plt.subplots(subplot_kw={'projection':'windrose'})
ax.bar(dfwD['1'],dfwM['1'], normed=True,edgecolor='white' ,cmap=cm.cool)
ax.set_legend()
plt.suptitle('Southwest Field Bottom Layer Current Rose')
plt.savefig('Southwest_Bottom_Layer_Rose_.png')

fig, ax=plt.subplots(subplot_kw={'projection':'windrose'})
ax.bar(dfwD['9'],dfwM['9'], normed=True,edgecolor='white' ,cmap=cm.cool)
ax.set_legend()
plt.suptitle('Southwest Field Middle Layer Current Rose')
plt.savefig('Southwest_Middle_Layer_Rose_.png')

fig, ax=plt.subplots(subplot_kw={'projection':'windrose'})
ax.bar(dfwD['16'],dfwM['16'], normed=True,edgecolor='white' ,cmap=cm.cool)
ax.set_legend()
plt.suptitle('Southwest Field Upper Layer Current Rose')
plt.savefig('Southwest_Upper_Layer_Rose_.png')

# %% plot observed windrose


wind=pd.read_csv('ISAP_Hour_wind.csv')
wind['WINDDIRECTION']=270-wind['WINDDIRECTION']
wind[wind['WINDDIRECTION']<0]['WINDDIRECTION']=wind[wind['WINDDIRECTION']<0]['WINDDIRECTION']+360
wind[wind['WINDSPEED']>3]['WINDDIRECTION'].median()

fig, ax=plt.subplots(subplot_kw={'projection':'windrose'})
ax.bar(wind['WINDDIRECTION'],wind['WINDSPEED'], normed=True,edgecolor='white' ,cmap=cm.cool)
ax.set_legend()
plt.suptitle('Wind Rose ISAP')
plt.savefig('Plot_Wind_Rose_.png')

# %%


