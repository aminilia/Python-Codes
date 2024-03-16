# %% 

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation 
from matplotlib.dates import DateFormatter
from PIL import Image
from datetime import datetime

# %%

dpe=pd.read_csv('Depth_east.csv')
dpm=pd.read_csv('Depth_main.csv')
dpw=pd.read_csv('Depth_west.csv')

dfe=pd.read_csv('East_ADCP_CSA.csv', header=[0,1], index_col=[0] )        # east 11
dfm=pd.read_csv('Main_ADCP_CSA.csv', header=[0,1], index_col=[0] )        # main 12
dfw=pd.read_csv('West_ADCP_CSA.csv', header=[0,1], index_col=[0] )        # west 17

# %% 

dfe.index = pd.to_datetime(dfe.index)
dfm.index = pd.to_datetime(dfm.index)
dfw.index = pd.to_datetime(dfw.index)

# %% 

dfe=dfe.dropna()
dfm=dfm.dropna()
dfw=dfw.dropna()

# %% 

dfeM=dfe['Mag']
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

# %% remove gap in data

fig, ax = plt.subplots()

def update(frame):
    ax.clear()
    ax.grid(True)
    ax.set(xlim=[-1.0,1.0])
    ax.set(ylim=[-13,0])
    ax.plot(dfeu.iloc[frame],-dpe['Depth Below Surface'])
    ax.set(xlabel='Velocity')
    ax.set(ylabel='Depth')
#    ax.xaxis.set_major_formatter('%Y-%m-%d')

# %% 

animation = FuncAnimation(fig, update, frames=len(dfeu), interval=1, blit=False)

# %% 

animation.save('df_East_Field_u_anim.gif', writer='pillow', fps=1)

# %% 


