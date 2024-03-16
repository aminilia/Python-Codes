# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 08:39:56 2023

@author: Emily.Day
"""


import os
import numpy as np
import xarray as xr
import netCDF4
import pandas as pd
import datetime as dt
from scipy.io import savemat
import scipy.io

#%% Load one doppio file to obatin the loctions of where the nodes are in netcdf used in detrmining indices
os.chdir(r'E:\219522-Offshore_Wind_Impact_BOEM\ROMS_data')
nc=netCDF4.Dataset('data_2017-11-01_2017-11-06_v3.nc')  #ROMS data
# print(nc.variables)
lat=np.asarray(nc.variables['lat_rho'])
lat_round=np.round(lat,8)
lon=np.asarray(nc.variables['lon_rho'])
lon_round=np.round(lon,8)
#%% Load the excel of closest doppio node to boundary nodes
os.chdir(r'E:\219522-Offshore_Wind_Impact_BOEM\Boundary\Temp_Salinity')
data_roms_match=pd.read_excel('Doppio_Match_Closest_Distance_BC_Nodes_07_24_2023_meters.xlsx')
#Data Doppio related to bc nodes model
lat_match=data_roms_match['doppio_lat'].to_numpy()
lat_match_round=np.round(lat_match,8)
lon_match=data_roms_match['doppio_lon'].to_numpy()
lon_match_round=np.round(lon_match,8)
#bc nodes model
bc_node_x=data_roms_match['lon_bc'].to_numpy()
bc_node_x_round=np.round(bc_node_x,8)
bc_node_y=data_roms_match['lat_bc'].to_numpy()
bc_node_y_round=np.round(bc_node_y,8)

#%%  Find Indices of doppio data in for loop
index_x_all=[]
index_y_all=[]
for i in range (0,len(lon_match_round)):
    index_lat=np.where(lat_match_round[i]==lat_round)
    index_lon=np.where(lon_match_round[i]==lon_round)
    if index_lat==index_lon:
        # print('True')
        index_x=index_lat[0][0]
        index_x_all.append(index_x)
        index_y=index_lon[1][0]
        index_y_all.append(index_y)
        
#%% Load the Delft3dFM Boundary PLI information
os.chdir(r"E:\219522-Offshore_Wind_Impact_BOEM\Boundary\Temp_Salinity")
boundary_nodes_pli=pd.read_excel('Boundary_BOEM_Node_Numbers_PLI_file_Updated.xlsx')
node_numbers=np.asarray(boundary_nodes_pli['Node_Number'])
bc_pli_lon=np.asarray(boundary_nodes_pli['Long'])
data_bc_pli_lon_round=np.round(bc_pli_lon,8)
bc_pli_lat=np.asarray(boundary_nodes_pli['Lat'])
data_bc_pli_lat_round=np.round(bc_pli_lat,8)

#%% Create list of all the ROMS data files

# os.chdir(r'E:\219522-Offshore_Wind_Impact_BOEM\Boundary\Current_Input')
os.chdir(r'E:\219522-Offshore_Wind_Impact_BOEM\ROMS_data\Current\data\Velocity_Rho_NanMean')
mat_file_name_vel='velocity_01_2018_ux_uy_nanmean'
mat_vel=scipy.io.loadmat(mat_file_name_vel)

vel_lat=mat_vel['lat_rho']
vel_lat_round=np.round(vel_lat,8)
vel_lon=mat_vel['lon_rho']
vel_lon_round=np.round(vel_lon,8)
u_vel=mat_vel['u_rotated']
v_vel=mat_vel['v_rotated']
    

#%% Load the water level lat lon indices they are not the same numbering as the boundary pli numbering
wl_boundary_nodes_v1=pd.read_excel('E:/219522-Offshore_Wind_Impact_BOEM/Boundary/Grid_Coordinates_Node_BOEM_Spherical_Subset_Save2.xlsx')
wl_boem_field_num=wl_boundary_nodes_v1['Field1']
wl_boem_grid_lon=np.asarray(wl_boundary_nodes_v1['Node_X'])
wl_data_boem_grid_lon=np.round(wl_boem_grid_lon,8)
wl_boem_grid_lat=np.asarray(wl_boundary_nodes_v1['Node_Y'])
wl_data_boem_grid_lat=np.round(wl_boem_grid_lat,8)


        
##########################################################################################################################################################################    


z_layers_int_all=[]
issue_all=[]
for i in range(0,len(index_x_all)):
    ind_x=index_x_all[i] 
    ind_y=index_y_all[i]
    
    #find which boundary node number is running through the for loop and save the bc node number index
    bc_node_x_val=bc_node_x_round[i]
    bc_node_y_val=bc_node_y_round[i]
    index_bc_pli=np.where(bc_node_x_val==data_bc_pli_lon_round) and np.where(bc_node_y_val==data_bc_pli_lat_round)
    index_bc_pli_n=index_bc_pli[0][0]
    bc_node_number_index=node_numbers[index_bc_pli_n]
    
    #load water level data that matches the boundary node
    index_bc_wl_file=np.where(bc_node_x_val==wl_data_boem_grid_lon) and np.where(bc_node_y_val==wl_data_boem_grid_lat)
    index_bc_wl_file_n=index_bc_wl_file[0][0]
    wl_number_file=str(wl_boem_field_num[index_bc_wl_file_n])
    mat_file_name='E:/219522-Offshore_Wind_Impact_BOEM/Boundary/Tidal/files/wl_files_p4/BOEM_WL_Weighted_BC_node_'+wl_number_file+'_p4.mat'
    mat=scipy.io.loadmat(mat_file_name)
    wl_bc_lat=mat['bc_lat'][0][0]
    data_bc_lat_round=np.round(wl_bc_lat,8)
    wl_bc_lon=mat['bc_lon'][0][0]
    data_bc_lon_round=np.round(wl_bc_lon,8)
    if bc_node_x_val==data_bc_lon_round and bc_node_y_val==data_bc_lat_round:
        print('node file='+str(bc_node_number_index))
    
    bc_water_level=mat['total_wl']
    avg_wl=np.mean(bc_water_level)
    zeta_ind=np.copy(avg_wl)    # using average weighted waterlevel at specific boundary node, then used in vertical stretching formulations below
    
    #load example doppio set to get depth layering
    os.chdir(r'E:\219522-Offshore_Wind_Impact_BOEM\ROMS_data')
    file_nc1='data_2017-11-01_2017-11-06_v3.nc'
    nc_file1=netCDF4.Dataset(file_nc1)
    
    s_w=np.asarray(nc_file1.variables['s_w'])                      # S-Coordinate at RHO-points
    hc=np.asarray(nc_file1.variables['hc'])                        # S-Coordinate Critical Depth
    Cs_w=np.asarray(nc_file1.variables['Cs_w'])                    # S-Coordinate Stretching Curves at RHO-points
    V_Transform=np.asarray(nc_file1.variables['Vtransform'])       # Vertical Terrain-Following transformation equation
    h=np.asarray(nc_file1.variables['h'])
    h_ind=h[ind_x,ind_y]
    

    z_w_all=[]
    for iii in range(0,len(Cs_w)): #Calculates the depth at each layer interface
        Zo_w=(hc*s_w[iii]+Cs_w[iii]*h_ind)/(hc+h_ind)
        z_w=zeta_ind+(zeta_ind+h_ind)*Zo_w
        z_w_all.append(z_w)
    z_layers_int_all.append(z_w_all)
   
    rho_layer_all=[]
    for nn in range(0,len(Cs_w)-1): #Calculated the depth at each layer center
        depth_layer_1=z_w_all[nn]
        depth_layer_2=z_w_all[nn+1]
        diff_depth_layer=abs(depth_layer_2-depth_layer_1)
        rho_layer=(diff_depth_layer/2)+depth_layer_1
        rho_layer_all.append(rho_layer)
    rho_layer_percent=((zeta_ind-rho_layer_all)/(zeta_ind-(h_ind*-1)))*100  #percentage of rho layer from surface
    



    u_vel_val=u_vel[:,:,ind_x,ind_y]
    v_vel_val=v_vel[:,:,ind_x,ind_y]
    time=mat_vel['Time'][0,:]

 
    d=dt.datetime(2006,1,1,0,0)
    d3=[]
    for t in range(0,len(time)):
        time_int=time[t]
        d2=d+dt.timedelta(seconds=time_int)
        d3.append(d2)

        

#Save Mat file with 
#Temperature, Salinity, depth levels, depth percentage, time, doppio point lon, doppio point lat,
    os.chdir(r'E:\219522-Offshore_Wind_Impact_BOEM\Boundary\Current_Input\mat_files_all_nanmean')  
    mdic = {"time_array": time, "doppio_level_interface":z_w_all,"rho_layer_percent":rho_layer_percent,"rho_layer":rho_layer_all,
            'depth':h_ind, 'zeta_ind':zeta_ind,
            "final_ux_array":u_vel_val,"final_uy_array": v_vel_val,
            'bc_node_x':bc_node_x_val,'bc_node_y':bc_node_y_val} 
    save_mat_name='velocity_ux_uy'+bc_node_number_index+'.mat'
    savemat(save_mat_name,mdic)  # Then save each as boundary node mat or nc file to be pulled into the depth layer file.
                   

#####################################################################3
# Obtained snippet of code from: https://docs.xarray.dev/en/stable/examples/ROMS_ocean_model.html
# if V_Transform == 1:
#     Zo_rho = hc * (s_rho - Cs_r) + Cs_r * h
#     z_rho = Zo_rho + zeta * (1 + Zo_rho / h)
# elif V_Transform == 2:
#     Zo_rho = (hc * s_rho + Cs_r * h) / (hc + h)
#     z_rho = zeta + (zeta + h) * Zo_rho

# ds.coords["z_rho"] = z_rho.transpose()  # needing transpose seems to be an xarray bug
# ds.salt
# z_rho_all=[]
# for i in range(0,len(Cs_r)):
#     print(i)
#     Zo_rho=(hc*s_rho[i]+Cs_r[i]*h[27,54])/(hc+h[27,54])
#     z_rho=zeta[0,27,54]+(zeta[0,27,54]+h[27,54])*Zo_rho
#     z_rho_all.append(z_rho)
