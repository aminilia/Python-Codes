# %%
##########################################
##############Amin Ilia###################
##############3/14/2024###################
##########################################
########program to plot histograms########
##########################################
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
dfhgt['bin']=pd.cut(dfhgt.index, bins=bins, labels=[bins[i] for i in range(len(bins) - 1)])
dfhgt['bin']=dfhgt['bin'].astype(str).fillna('> 200')
dfhgt=dfhgt.apply(pd.to_numeric, errors='coerce')
mean_bt=dfhgt.groupby('bin').mean()
mean_bt.index=pd.to_numeric(mean_bt.index)
mean_bt=mean_bt.sort_index()
# %% 
no_model=12
dfsc=pd.DataFrame()
dfst=pd.DataFrame()
for i in range(no_model):
    dff=pd.read_csv('C:\\work\\BOEM\\Task 4\\Validation\\Baroclinic_Run_Valid\\Baroclinic_Run'+str(i+1)+'\\temp_bias_yearly.xyz',delimiter='\s+',names=['x','y','depth',i+1],index_col=2)
    dff.drop(['x','y'],inplace=True,axis=1)
    dfst=pd.concat([dfst,dff],axis=1)
    dff=pd.read_csv('C:\\work\\BOEM\\Task 4\\Validation\\Baroclinic_Run_Valid\\Baroclinic_Run'+str(i+1)+'\\current_bias_yearly.xyz',delimiter='\s+',names=['x','y','depth',i+1],index_col=2)
    dff.drop(['x','y'],inplace=True,axis=1)
    dfsc=pd.concat([dfsc,dff],axis=1)
dfst.index=-dfst.index
dfst['bin']=pd.cut(dfst.index, bins=bins, labels=[bins[i] for i in range(len(bins) - 1)])
dfst['bin']=dfst['bin'].astype(str).fillna('> 200')
dfst=dfst.apply(pd.to_numeric, errors='coerce')
mean_bst=dfst.groupby('bin').mean()
mean_bst.index=pd.to_numeric(mean_bst.index)
mean_bst=mean_bst.sort_index()
dfsc.index=-dfsc.index
dfsc['bin']=pd.cut(dfsc.index, bins=bins, labels=[bins[i] for i in range(len(bins) - 1)])
dfsc['bin']=dfsc['bin'].astype(str).fillna('> 200')
dfsc=dfsc.apply(pd.to_numeric, errors='coerce')
mean_bsc=dfsc.groupby('bin').mean()
mean_bsc.index=pd.to_numeric(mean_bsc.index)
mean_bsc=mean_bsc.sort_index()
# %% drop model 3
model_listd=model_list[:2]+model_list[3:]
mean_bs.drop((3),axis=1,inplace=True)
mean_bt.drop((3),axis=1,inplace=True)
mean_bst.drop((3),axis=1,inplace=True)
mean_bsc.drop((3),axis=1,inplace=True)
# %% 
bin_labels=[f'({bins[i]}-{bins[i+1]})' for i in range(len(bins) - 1)]
bin_labels[10]=(r'($\geq$200)')
# %% 
os.chdir('C:\\work\\BOEM\\Task 4\\Validation\\Histogram')
fig = plt.figure(figsize=(12, 14))
gs = GridSpec(2, 1, height_ratios=[1, 1])
bar_width=1.0
ax1 = fig.add_subplot(gs[0])
for i in mean_bs.columns:
    ax1.bar(bins[:-1]+1*i-len(model_listd)/2*bar_width, abs(mean_bs[i]), width=bar_width, label='Model_ID'+str(i).zfill(2))
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
table = ax2.table(cellText=tmeans.round(2).abs().astype(str).values.tolist(), rowLabels=model_listd, colLabels=bin_labels, loc='upper center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(13)  # Adjusted font size to fit the table cells
table.scale(1, 2)  # Scale table size for better readability
plt.tight_layout()
plt.subplots_adjust(hspace=0.10)  # Reduce space between histogram and table
plt.savefig('hist_sal_with_table_e3.png',dpi=300)
# %% 
fig = plt.figure(figsize=(12, 14))
gs = GridSpec(2, 1, height_ratios=[1, 1])
bar_width=1.0
ax1 = fig.add_subplot(gs[0])
for i in mean_bt.columns:
    ax1.bar(bins[:-1]+1*i-len(model_listd)/2*bar_width, abs(mean_bt[i]), width=bar_width, label='Model_ID'+str(i).zfill(2))
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
table = ax2.table(cellText=tmeant.round(2).abs().astype(str).values.tolist(), rowLabels=model_listd, colLabels=bin_labels, loc='upper center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(13)  # Adjusted font size to fit the table cells
table.scale(1, 2)  # Scale table size for better readability
plt.tight_layout()
plt.subplots_adjust(hspace=0.10)  # Reduce space between histogram and table
plt.savefig('hist_temp_with_table_e3.png',dpi=300)
# %% 
fig = plt.figure(figsize=(12, 14))
gs = GridSpec(2, 1, height_ratios=[1, 1])
bar_width=1.0
ax1 = fig.add_subplot(gs[0])
for i in mean_bsc.columns:
    ax1.bar(bins[:-1]+1*i-len(model_listd)/2*bar_width, abs(mean_bsc[i]), width=bar_width, label='Model_ID'+str(i).zfill(2))
ax1.set(xticks=bins[:-1], xticklabels=bin_labels)
ax1.set_xlabel('Bathymetry bins (m)', fontsize=14)
ax1.tick_params(axis='both', which='major', labelsize=12)
plt.ylim(0,0.3)
ax1.set_ylabel('mean Current Bias for Codar (m/s)', fontsize=14)
ax1.legend(loc='upper left')
# Add a table directly below the histograms
ax2 = fig.add_subplot(gs[1])
ax2.axis('off')
tmeant=mean_bsc.transpose()
# Create the table
table = ax2.table(cellText=tmeant.round(2).abs().astype(str).values.tolist(), rowLabels=model_listd, colLabels=bin_labels, loc='upper center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(13)  # Adjusted font size to fit the table cells
table.scale(1, 2)  # Scale table size for better readability
plt.tight_layout()
plt.subplots_adjust(hspace=0.10)  # Reduce space between histogram and table
plt.savefig('hist_surf_current_with_table_e3.png',dpi=300)
# %% 
binst=np.delete(bins,-3,axis=0)
bin_labelst=bin_labels[:-2]+bin_labels[-1:]
fig = plt.figure(figsize=(12, 14))
gs = GridSpec(2, 1, height_ratios=[1, 1])
bar_width=1.0
ax1 = fig.add_subplot(gs[0])
for i in mean_bst.columns:
    ax1.bar(binst[:-1]+1*i-len(model_listd)/2*bar_width, abs(mean_bst[i]), width=bar_width, label='Model_ID'+str(i).zfill(2))
ax1.set(xticks=binst[:-1], xticklabels=bin_labelst)
ax1.set_xlabel('Bathymetry bins (m)', fontsize=14)
ax1.tick_params(axis='both', which='major', labelsize=12)
plt.ylim(0,1.1)
ax1.set_ylabel('mean Surface Temperature Bias for SST (°C)', fontsize=14)
ax1.legend(loc='upper left')
# Add a table directly below the histograms
ax2 = fig.add_subplot(gs[1])
ax2.axis('off')
tmeant=mean_bst.transpose()
# Create the table
table = ax2.table(cellText=tmeant.round(2).abs().astype(str).values.tolist(), rowLabels=model_listd, colLabels=bin_labelst, loc='upper center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(13)  # Adjusted font size to fit the table cells
table.scale(1, 2)  # Scale table size for better readability
plt.tight_layout()
plt.subplots_adjust(hspace=0.10)  # Reduce space between histogram and table
plt.savefig('hist_surf_temp_with_table_e3.png',dpi=300)
# %% 