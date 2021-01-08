import os
import glob

nii_list = glob.glob("./*.nii.gz")
nii_list.sort()

for nii_path in nii_list:
    cmd = "gunzip " + nii_path
    print(cmd)
    os.system(cmd)