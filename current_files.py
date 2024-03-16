# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 00:14:03 2023

@author: Emily.Day
"""

import os
import numpy as np
import netCDF4
import pandas as pd
import datetime as dt
import scipy.io


# Python Script to create boundary water level file. "WaterLevel.bc"


d=dt.datetime(2017,11,1,0,0)
d3=[]
# d2=d
# d3.append(d2)
t_init=int(-3600)
t_sec_all=[]
for t in range(0,22000):
    t_sec=int(t_init+3600)
    t_init=int(t_sec)
    d2=d+dt.timedelta(seconds=t_init)

    d3.append(d2)
    t_sec_all.append(t_sec)


   

time_array=np.array(t_sec_all)
time_array_str=time_array.astype('str')


os.chdir(r"E:\219522-Offshore_Wind_Impact_BOEM\Boundary\Temp_Salinity")
boundary_nodes_pli=pd.read_excel('Boundary_BOEM_Node_Numbers_PLI_file_Updated.xlsx')

node_numbers=np.asarray(boundary_nodes_pli['Node_Number'])
bc_pli_lon=np.asarray(boundary_nodes_pli['Long'])
data_bc_pli_lon_round=np.round(bc_pli_lon,8)
bc_pli_lat=np.asarray(boundary_nodes_pli['Lat'])
data_bc_pli_lat_round=np.round(bc_pli_lat,8)


# interval=0
interval=1464
interval_vel=0
for n in range(0,15):
    os.chdir(r"E:\219522-Offshore_Wind_Impact_BOEM\Boundary\Current_Input\Boundary_Files")
    time_int1=int(interval)
    time_int2=time_int1+73
    time_int_vel=int(interval_vel)
    time_int_vel2=time_int_vel+73
    # time_int2=time_int1+3
    d3_subset=d3[time_int1:time_int2]
    interval=int(time_int2)-1
    interval_vel=int(time_int_vel2)-1
    
    time_array_subset=time_array_str[time_int1:time_int2]
    length_char=len(time_array_subset[len(time_array_subset)-1])
    format_string='{:'+str(length_char)+'}  '
    time_int1_str=str(d3_subset[0])
    
    # folder_name="Run_"+time_int1_str[0:4]+'_'+time_int1_str[5:7]+'_'+time_int1_str[8:10]
    folder_name="Run_"+time_int1_str[0:4]+'_'+time_int1_str[5:7]+'_'+time_int1_str[8:10]+'_'+time_int1_str[11:13]+'_'+time_int1_str[14:16]
    os.mkdir(folder_name)
    os.chdir(folder_name)
    f=open("uxuyadvection_test.bc","w+")

    for i in range(0,len(node_numbers)):
        bc_pli_lon_val=data_bc_pli_lon_round[i]
        bc_pli_lat_val=data_bc_pli_lat_round[i]
        node_num_int=node_numbers[i]
        mat_file_name='E:\\219522-Offshore_Wind_Impact_BOEM\Boundary\Current_Input\mat_files_all_nanmean\\velocity_ux_uy'+node_num_int+'.mat'
        mat=scipy.io.loadmat(mat_file_name)
        bc_lat_mat=mat['bc_node_y'][0][0]
        data_bc_lat_round=np.round(bc_lat_mat,8)
        bc_lon_mat=mat['bc_node_x'][0][0]
        data_bc_lon_round=np.round(bc_lon_mat,8)


        if bc_pli_lon_val==data_bc_lon_round and bc_pli_lat_val==data_bc_lat_round:
            ux_array=mat["final_ux_array"][time_int_vel:time_int_vel2]
            uy_array=mat["final_uy_array"][time_int_vel:time_int_vel2]
            grid_depth_percent=mat["rho_layer_percent"][0]
            grid_depth_percent_2=100-grid_depth_percent # subtract grid depth percent to get it to be percentage from bedlevel not from surface.Basically re orienting percent
            

            grid_depth_percent_cumulative_len=len(grid_depth_percent_2)
            grid_depth_percent_cumulative_v2_1=np.copy(grid_depth_percent_2)
            grid_depth_percent_cumulative_v2_1[0]=0
            grid_depth_percent_cumulative_v2_1[grid_depth_percent_cumulative_len-1]=100
            grid_depth_percent_cumulative_v2=grid_depth_percent_cumulative_v2_1.round(2)
    
            formatstring2=""
            formatstring3='{:'+str(length_char)+'}  '
            formatstring4=''
            for line in grid_depth_percent_cumulative_v2:
                formatstring2+="{:g} ".format(line)
                formatstring3+='{:.2f}  '  
                formatstring4+='{:.2f}%  ' 
    
            f.write('[forcing]\n')
            f.write('Name                            = '+node_num_int+'\n')
            f.write('Function                        = t3d\n')
            f.write('Time-interpolation              = linear\n')
            f.write('Vertical position type          = percentage from bed\n')
            f.write('Vertical position specification = '+formatstring2+'\n')
            f.write('Quantity                        = time\n')
            f.write('Unit                            = seconds since 2017-11-01 00:00:00\n')
            f.write('Vector                            = uxuyadvectionvelocitybnd:ux,uy\n')
            for ii in range(0, np.size(grid_depth_percent)):
                unit_number=str(ii+1)
    
                f.write('Quantity                        = ux\n')
                f.write('Unit                            = -')
                f.write('Vertical position               = '+unit_number+'\n')
                f.write('Quantity                        = uy\n')
                f.write('Unit                            = -')
                f.write('Vertical position               = '+unit_number+'\n')
    
            for t in range(0,len(time_array_subset)):
    
                ux_values=''
                uy_values=''
                ux_time=ux_array[t,0:ux_array.shape[1]]
                uy_time=uy_array[t,0:uy_array.shape[1]]
                for test in ux_time:
                    ux_values+="{:.4f}  ".format(test)
                for test2 in uy_time:
                    uy_values+="{:.4f}  ".format(test2)
    
                f.write(format_string.format(str(time_array_subset[t])))
                f.write(ux_values)
                f.write(uy_values+'\n')
            f.write('\n')
    f.close()