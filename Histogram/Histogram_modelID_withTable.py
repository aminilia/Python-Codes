# %%
#################
####Amin Ilia#### 
####3/14/2024####
#################

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import os 
import cftime
from matplotlib.gridspec import GridSpec

# %% 

os.chdir('C:\\work\\BOEM\\Task 4\\Validation\\Glider')
no_model=12
dfhgs=pd.DataFrame()
dfhgt=pd.DataFrame()
for i in range(no_model):
    dff=pd.read_csv('Bias_hist_s_ID{:02d}'.format(i+1)+'.csv',names=[i+1],skiprows=[0],index_col=0)
    dfhgs=pd.concat([dfhgs,dff],axis=1)
    dff=pd.read_csv('Bias_hist_t_ID{:02d}'.format(i+1)+'.csv',names=[i+1],skiprows=[0],index_col=0)
    dfhgt=pd.concat([dfhgt,dff],axis=1)

# %% 

bins=np.arange(0,221,20)

bins[11]=5000

model_list=['Model ID '+str(num).zfill(2) for num in range(1, 13)]

dfhgs['bin'] = pd.cut(dfhgs.index, bins=bins, labels=[bins[i] for i in range(len(bins) - 1)])

dfhgs['bin']=dfhgs['bin'].astype(str).fillna('> 200')
dfhgs=dfhgs.apply(pd.to_numeric, errors='coerce')

mean_bs=dfhgs.groupby('bin').mean()
mean_bs.index=pd.to_numeric(mean_bs.index)
mean_bs=mean_bs.sort_index()


dfhgt['bin'] = pd.cut(dfhgt.index, bins=bins, labels=[bins[i] for i in range(len(bins) - 1)])

dfhgt['bin']=dfhgt['bin'].astype(str).fillna('> 200')
dfhgt=dfhgt.apply(pd.to_numeric, errors='coerce')

mean_bt=dfhgt.groupby('bin').mean()
mean_bt.index=pd.to_numeric(mean_bt.index)
mean_bt=mean_bt.sort_index()

# %% 

os.chdir('C:\\work\\BOEM\\Task 4\\Validation\\Baroclinic_Run_Valid')
no_model=12
dfsc=pd.DataFrame()
dfst=pd.DataFrame()
for i in range(no_model):
    dff=pd.read_csv('Bias_hist_s_ID{:02d}'.format(i+1)+'.csv',names=[i+1],skiprows=[0],index_col=0)
    dfhgs=pd.concat([dfhgs,dff],axis=1)
    dff=pd.read_csv('Bias_hist_t_ID{:02d}'.format(i+1)+'.csv',names=[i+1],skiprows=[0],index_col=0)
    dfhgt=pd.concat([dfhgt,dff],axis=1)

# %% 

bin_labels=[f'({bins[i]}-{bins[i+1]})' for i in range(len(bins) - 1)]

bin_labels[10]=(r'($\geq$200)')

# %% 

os.chdir('C:\\work\\BOEM\\Task 4\\Validation\\Histogram')
fig = plt.figure(figsize=(12, 14))
gs = GridSpec(2, 1, height_ratios=[1, 1])
bar_width=1.0
ax1 = fig.add_subplot(gs[0])
for i in range(len(model_list)):
    ax1.bar(bins[:-1]+1*i-len(model_list)/2*bar_width, abs(mean_bs[i+1]), width=bar_width, label=model_list[i])

ax1.set(xticks=bins[:-1], xticklabels=bin_labels)
ax1.set_xlabel('Bathymetry bins (m)', fontsize=14)
ax1.tick_params(axis='both', which='major', labelsize=12)

plt.ylim(0,1.1)
ax1.set_ylabel('mean Salinity Bias for Glider (psu)', fontsize=14)
ax1.legend(loc='upper left')

# Add a table directly below the histograms
ax2 = fig.add_subplot(gs[1])
ax2.axis('off')
tmeans=mean_bs.transpose()
# Create the table
table = ax2.table(cellText=tmeans.round(2).abs().astype(str).values.tolist(), rowLabels=model_list, colLabels=bin_labels, loc='upper center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(13)  # Adjusted font size to fit the table cells
table.scale(1, 2)  # Scale table size for better readability
plt.tight_layout()
plt.subplots_adjust(hspace=0.10)  # Reduce space between histogram and table
plt.savefig('hist_sal_with_table.png',dpi=300)

# %% 

fig = plt.figure(figsize=(12, 14))
gs = GridSpec(2, 1, height_ratios=[1, 1])
bar_width=1.0
ax1 = fig.add_subplot(gs[0])
for i in range(len(model_list)):
    ax1.bar(bins[:-1]+1*i-len(model_list)/2*bar_width, abs(mean_bt[i+1]), width=bar_width, label=model_list[i])

ax1.set(xticks=bins[:-1], xticklabels=bin_labels)
ax1.set_xlabel('Bathymetry bins (m)', fontsize=14)
ax1.tick_params(axis='both', which='major', labelsize=12)

plt.ylim(0,3.4)
ax1.set_ylabel('mean Temperature Bias for Glider (°C)', fontsize=14)
ax1.legend(loc='upper left')

# Add a table directly below the histograms
ax2 = fig.add_subplot(gs[1])
ax2.axis('off')
tmeant=mean_bt.transpose()
# Create the table
table = ax2.table(cellText=tmeant.round(2).abs().astype(str).values.tolist(), rowLabels=model_list, colLabels=bin_labels, loc='upper center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(13)  # Adjusted font size to fit the table cells
table.scale(1, 2)  # Scale table size for better readability
plt.tight_layout()
plt.subplots_adjust(hspace=0.10)  # Reduce space between histogram and table
plt.savefig('hist_temp_with_table.png',dpi=300)

# %% 
