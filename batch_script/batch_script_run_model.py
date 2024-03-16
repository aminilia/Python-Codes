# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 17:37:23 2023

@author: Amin Ilia
"""

import os
import numpy as np
import netCDF4
import pandas as pd
import datetime as dt
import scipy.io

path_1=r'F:\BOEM\v12'         # change it to base path
path_2=r'F:\BOEM\v12\output_spatial_vari_Eddy_sd2_smo1_calib'      # outputs will be moved to this folder

ncpu=31                              # no. of CPUs

os.chdir(path_1+'\Runs')
f= open("run_"+str(ncpu)+"_baroclinic.bat","w+")

folder=os.listdir()
folder_name=[]

restart_name_list=[]
for g in os.listdir():
    if g.startswith("Run"):
        fname=g
        restart_name=fname[4:8]+fname[9:11]+fname[12:14]+'_'+fname[15:17]+fname[18:20]
        folder_name.append(fname)
        restart_name_list.append(restart_name)

change_runs_path='cd '+path_1+'\Runs\n'      
change_base_folder='cd '+path_1+'\Main\dflowfm\n'
change_wind_folder='cd '+path_1+'\wind\n'
change_xml_folder='cd '+path_1+'\Main\\xml\n'


int1=5270400
int1_1=5270400+259200

prev_restart_file_str='FlowFM_merged_20180101_000000_rst.nc'


for i in range(0,len(folder_name)):
    
    model_run_folder=folder_name[i]
    
    if i<len(folder_name)-1:
        next_model_run_folder=folder_name[i+1]
    else:
        next_model_run_folder=model_run_folder[0:12]+str(int(model_run_folder[12:14])+3).zfill(2)+model_run_folder[14:]
        print(next_model_run_folder)
    
    change_model_run_folder='cd '+path_1+'\Runs\\'+model_run_folder+'\n'
    restart_file_time=restart_name_list[i]
    change_output_path='cd '+path_1+'\Runs\\'+model_run_folder+'\output\n'
    
    if i>0:
        prev_model_run_folder=folder_name[i-1]
#        prev_restart_file_time=restart_name_list[i-1]
    else:
        prev_model_run_folder='Initial_Run'
#        prev_restart_file_time='20180101_0000'
        
    
    restart_file_str='FlowFM_merged_'+restart_file_time+'00_rst.nc'
#    prev_restart_file_str='FlowFM_merged_'+prev_restart_file_time+'00_rst.nc'


    change_STBC_folder='cd '+path_1+'\Boundary_Files_Temp_Salinity\\'+model_run_folder+'\n'
    change_WL_folder='cd '+path_1+'\Boundary_Files_Water_Level\\'+model_run_folder+'\n'

            
    if i>0:
#        start_val1=int1
        start_val2=int1+259200
        int1=int(start_val2)
#        stop_val1=int1_1
        stop_val2=int1_1+259200
        int1_1=int(stop_val2)
    else:
        start_val1=int1
        start_val2=int1
        int1=int(start_val2)
        stop_val1=int1_1
        stop_val2=int1_1
        int1_1=int(stop_val2)
        tstart1=str(start_val1)
        tstop1=str(stop_val1)

    tstart2=str(start_val2)
    tstop2=str(stop_val2)

    pshell_mdu_change1='powershell -Command "(gc FlowFM.mdu) -replace \'TStart='+tstart1+'\', \'TStart='+tstart2+'\' | Out-File -encoding ASCII FlowFM.mdu"\n'
    pshell_mdu_change2='powershell -Command "(gc FlowFM.mdu) -replace \'TStop='+tstop1+'\', \'TStop='+tstop2+'\' | Out-File -encoding ASCII FlowFM.mdu"\n'
    pshell_mdu_change3='powershell -Command "(gc FlowFM.mdu) -replace \'RestartFile='+prev_restart_file_str+'\', \'RestartFile='+restart_file_str+'\' | Out-File -encoding ASCII FlowFM.mdu"\n'
    restart_file_path=path_1+'\\Runs\\'+prev_model_run_folder+'\\output\\'+restart_file_str

    pshell_xml_change='powershell -Command "(gc model.xml) -replace \'<workingDir>Run_2018_01_01_00_00</workingDir>\', \'<workingDir>'+model_run_folder+'</workingDir>\' | Out-File -encoding ASCII model.xml"\n'

    pshell_ext_change1='powershell -Command "(gc FlowFM.ext) -replace \'FILENAME=windx_20180101_20180104.amu\', \'FILENAME=windx_'+model_run_folder[4:8]+model_run_folder[9:11]+model_run_folder[12:14]+'_'+next_model_run_folder[4:8]+next_model_run_folder[9:11]+next_model_run_folder[12:14]+'.amu\' | Out-File -encoding ASCII FlowFM.ext"\n'
    pshell_ext_change2='powershell -Command "(gc FlowFM.ext) -replace \'FILENAME=windy_20180101_20180104.amv\', \'FILENAME=windy_'+model_run_folder[4:8]+model_run_folder[9:11]+model_run_folder[12:14]+'_'+next_model_run_folder[4:8]+next_model_run_folder[9:11]+next_model_run_folder[12:14]+'.amv\' | Out-File -encoding ASCII FlowFM.ext"\n'

    f.write(change_model_run_folder)

    f.write('del /F/Q FlowFM_*.mdu\n')
    f.write('del /F/Q Main_Grid_and_Grid_Extension_Spherical_final_10_10_update_*.nc\n')
    f.write('move /y output old_output\n')
    
    
    
    f.write(change_base_folder)
    f.write('copy /v/y *.* '+path_1+'\Runs\\'+ model_run_folder+'\n')
    
    f.write(change_wind_folder)
    f.write('copy /v/y windx_'+model_run_folder[4:8]+model_run_folder[9:11]+model_run_folder[12:14]+'_*.amu '+path_1+'\Runs\\'+ model_run_folder+'\n')
    f.write('copy /v/y windy_'+model_run_folder[4:8]+model_run_folder[9:11]+model_run_folder[12:14]+'_*.amv '+path_1+'\Runs\\'+ model_run_folder+'\n')
    
    f.write(change_STBC_folder)
    f.write('copy /v/y *.* '+path_1+'\Runs\\'+ model_run_folder+'\n')
    
    f.write(change_WL_folder)
    f.write('copy /v/y *.* '+path_1+'\Runs\\'+ model_run_folder+'\n')

    f.write(change_model_run_folder)
    f.write(pshell_mdu_change1)
    f.write(pshell_mdu_change2)
    f.write(pshell_mdu_change3)
    
    f.write(pshell_ext_change1)
    f.write(pshell_ext_change2)


    f.write('call "C:\Program Files\Deltares\Delft3D FM Suite 2023.02 HMWQ\plugins\DeltaShell.Dimr\kernels\\x64\dflowfm\scripts\\run_dflowfm.bat" "--partition:ndomains='+str(int(ncpu))+':icgsolver=6" FlowFM.mdu\n')

    f.write(change_xml_folder)
    f.write('copy /v/y *.* '+path_1+'\Runs\n')
    
    f.write(change_runs_path)
    f.write(pshell_xml_change)
    
    f.write('echo "Model Run Start"\n')
    f.write('call "C:\Program Files\Deltares\Delft3D FM Suite 2023.02 HMWQ\plugins\DeltaShell.Dimr\kernels\\x64\dimr\scripts\\run_dimr_parallel.bat" '+str(int(ncpu))+' model.xml >> log_'+restart_name_list[i]+'.txt\n')
    
    if i<len(folder_name)-1:
        f.write(change_output_path)
        f.write('dir /b *'+restart_name_list[i+1]+'00_rst.nc >> list_rst.txt"\n')
        f.write('call "C:\Program Files\Deltares\Delft3D FM Suite 2023.02 HMWQ\plugins\DeltaShell.Dimr\kernels\\x64\dflowfm\scripts\\run_dfmoutput.bat" mapmerge --listfile list_rst.txt \n')
        f.write('copy /v/y *merged* '+path_1+'\Runs\\'+folder_name[i+1]+'\n')
        f.write('dir /b *_map.nc >> list_map.txt"\n')
        f.write('call "C:\Program Files\Deltares\Delft3D FM Suite 2023.02 HMWQ\plugins\DeltaShell.Dimr\kernels\\x64\dflowfm\scripts\\run_dfmoutput.bat" mapmerge --listfile list_map.txt \n')
        f.write('del /F/Q FlowFM_00*_map.nc\n')
        f.write('del /F/Q FlowFM_00*_rst.nc\n')
        f.write(change_model_run_folder)
        f.write('move /y output '+path_2+'\output_'+model_run_folder+'\n')
        f.write('del /F/Q *.*\n')


    else:
        f.write(change_output_path)
        f.write('dir /b *_map.nc >> list_map.txt"\n')
        f.write('call "C:\Program Files\Deltares\Delft3D FM Suite 2023.02 HMWQ\plugins\DeltaShell.Dimr\kernels\\x64\dflowfm\scripts\\run_dfmoutput.bat" mapmerge --listfile list_map.txt \n')
        f.write('del /F/Q FlowFM_00*_map.nc\n')
        f.write('del /F/Q FlowFM_00*_rst.nc\n')
        f.write(change_model_run_folder)
        f.write('move /y output '+path_2+'\output_'+model_run_folder+'\n')
        f.write('del /F/Q *.*\n')

        f.write('echo "Done!"\n')
        f.write('echo "Shutdown system"\n')
        f.write('Shutdown /s\n')

f.close()