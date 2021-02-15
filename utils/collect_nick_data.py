import glob
import os

source_path = "/data_local/nick/nifti/"
MRI_name = "Ax_T1_BRAVO_Stealth_CT"
PET_name = "PET_MAC_ZTE"
name_type = ".nii.gz"

MRI_list = glob.glob(source_path+"*/"+MRI_name+name_type)
PET_list = glob.glob(source_path+"*/"+PET_name+name_type)
MRI_list.sort()
PET_list.sort()

for MRI_data in MRI_list:
	print(MRI_data[33:])

for PET_data in PET_list:
	print(PET_data[33:])
