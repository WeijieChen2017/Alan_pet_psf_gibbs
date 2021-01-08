import os
import glob

nii_list = glob.glob("./*.nii")
nii_list.sort()

for nii_path in nii_list:
    cmd = "gzip " + nii_path
    print(cmd)
    os.syetem(cmd)