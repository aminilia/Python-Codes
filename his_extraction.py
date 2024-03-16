# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 17:12:29 2024

@author: aznamswan08.swan
"""

# %% 

import os
import numpy as np
import pandas as pd
import datetime as dt
import xarray as xr
import netCDF4
from matplotlib import rc
import matplotlib.pyplot as plt

# %%
main_path='F:\\BOEM\\v12_Bartrop\\output_barotropic_saved'
os.chdir(main_path)
output_folder_list=os.listdir()
Ux_all=[]
Uy_all=[]
time_all=[]
z_coor_all=[]

for i in range(1,len(output_folder_list)):
    file_path=main_path+'/'+output_folder_list[i]+'/FlowFM_0000_his.nc'
    if (output_folder_list[i]).startswith('output')==False:
        continue
    nc_file=netCDF4.Dataset(file_path)
    if i==1:
        
        face_x=np.asarray(nc_file.variables['station_geom_node_coordx'])
        face_y=np.asarray(nc_file.variables['station_geom_node_coordy'])
        
        z_coor=np.asarray(nc_file.variables['zcoordinate_c'])

        st_name=np.asarray(nc_file.variables['station_name'])
        
        Ux=np.asarray(nc_file.variables['x_velocity'])
        Uy=np.asarray(nc_file.variables['y_velocity'])
        
        time=np.asarray(nc_file.variables['time'])

        z_coor_all.append(z_coor) 
        time_all.append(time)    
        Ux_all.append(Ux)
        Uy_all.append(Uy)
        
    else:
        z_coor=np.asarray(nc_file.variables['zcoordinate_c'][1:,:,:])

        Ux=np.asarray(nc_file.variables['x_velocity'][1:,:,:])
        Uy=np.asarray(nc_file.variables['y_velocity'][1:,:,:])
        
        time=np.asarray(nc_file.variables['time'][1:])
        
        time_all.append(time)
        Ux_all.append(Ux)
        Uy_all.append(Uy)  
        z_coor_all.append(z_coor)
        
Ux_all_array=np.concatenate(Ux_all)
Uy_all_array=np.concatenate(Uy_all)
time_all_array=np.concatenate(time_all)
zcoor_all_array=np.concatenate(z_coor_all)

os.chdir('F:\\BOEM\\v12_Bartrop')
test_nc = netCDF4.Dataset('UV_his_barotropic.nc','w')

#%% NETCDF File Information
test_nc.institution='Deltares'
test_nc.references='http://www.deltares.nl'
test_nc.source='Deltares, D-Flow FM Version 1.2.177.142431, Jan 26 2023, 11:34:53, model'
test_nc.history='Created on 2023-10-10T21:32:55-0000, D-Flow FM'
test_nc.date_created='2023-10-10T21:32:55-0000'
test_nc.date_modified='2023-10-10T21:32:55-0000'
test_nc.Conventions='CF-1.5 Deltares-0.1'
test_nc.uuid='c963ee5f-ae3f-6747-ba44-eb92dfe1ed78'

#%% NETCDF File Set Dimensions

test_nc.createDimension('name_len',256)
test_nc.createDimension('station_geom_nNodes',31)
test_nc.createDimension('time',None)
test_nc.createDimension('laydim',40)
test_nc.createDimension('laydimw',41)
test_nc.createDimension('stations',31)

#%%# time
time_val = test_nc.createVariable('time',float,('time'))
time_val.units='seconds since 2017-11-01 00:00:00 +00:00'
time_val.standard_name='time'
time_val[:]=time_all_array

#%% #sea_surface_current_x
ux_val = test_nc.createVariable('x_velocity',float,('time','stations','laydim'))
ux_val.coordinates='station_x_coordinate station_y_coordinate station_name zcoordinate_c'
ux_val.standard_name='eastward_sea_water_velocity'
ux_val.long_name='flow element center velocity vector, x-component'
ux_val.units='m/s'
ux_val.grid_mapping='wgs84'
ux_val[:]=Ux_all_array

#%% #sea_surface_current_y
uy_val = test_nc.createVariable('y_velocity',float,('time','stations','laydim'))
uy_val.coordinates='station_x_coordinate station_y_coordinate station_name zcoordinate_c'
uy_val.standard_name='nothward_sea_water_velocity'
uy_val.long_name='flow element center velocity vector, x-component'
uy_val.units='m/s'
uy_val.grid_mapping='wgs84'
uy_val[:]=Uy_all_array

# %% #Station_name

stations_names = test_nc.createVariable('station_name','S1', ('stations','name_len'))
stations_names.cf_role='timeseries_id'
stations_names.long_name='observation station name'
stations_names[:]=st_name

#%% #Longitude, X-Coordinate of stations
x_flow_circumcenter_val = test_nc.createVariable('station_geom_node_coordx',float,('station_geom_nNodes'))
x_flow_circumcenter_val.units='m'
x_flow_circumcenter_val.standard_name='projection_x_coordinate'
x_flow_circumcenter_val.long_name='x-coordinate of station'
x_flow_circumcenter_val[:]=face_x

#%% #Latitude, Y-Coordinate of stations
y_flow_circumcenter_val = test_nc.createVariable('station_geom_node_coordy',float,('station_geom_nNodes'))
y_flow_circumcenter_val.units='m'
y_flow_circumcenter_val.standard_name='projection_y_coordinate'
y_flow_circumcenter_val.long_name='y-coordinate of station'
y_flow_circumcenter_val[:]=face_y

#%% z of layers center
z_flow_circumcenter_val = test_nc.createVariable('zcoordinate_c',float,('time','stations','laydim'))
z_flow_circumcenter_val.coordinates='station_x_coordinate station_y_coordinate station_name zcoordinate_c'
z_flow_circumcenter_val.units='m'
z_flow_circumcenter_val.standard_name='projection_y_coordinate'
z_flow_circumcenter_val.long_name='vertical coordinate at center of flow element and layer'
z_flow_circumcenter_val.grid_mapping='wgs84'
z_flow_circumcenter_val.positive='up'
z_flow_circumcenter_val[:]=zcoor_all_array

test_nc.close()