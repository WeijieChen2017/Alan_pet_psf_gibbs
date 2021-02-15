import glob
import os

source_path = "/data_local/nick/nifti/"
MRI_name = "Ax_T1_BRAVO_Stealth"
PET_name = "PET_MAC_ZTE"
name_type = ".nii.gz"

MRI_list = glob.glob(source_path+"*/"+MRI_name+name_type)
PET_list = glob.glob(source_path+"*/"+PET_name+name_type)
MRI_list.sort()
PET_list.sort()

for PET_data in PET_list:
    print("&"*50)
    print(PET_data)
    MRI_data = PET_data[:33]+MRI_name+name_type
    print(MRI_data)
    if MRI_data in MRI_list:
        print("Both exist.")

    cmd_PET = ("cp "+PET_data+" ./sCT_"+PET_data[29:32]+"_PET"+name_type)
    cmd_MRI = ("cp "+MRI_data+" ./sCT_"+MRI_data[29:32]+"_MRI"+name_type)
    print(cmd_PET)
    print(cmd_MRI)
    os.system(cmd_PET)
    os.system(cmd_MRI)