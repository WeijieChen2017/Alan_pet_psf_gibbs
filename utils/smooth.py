import os
import glob
import nibabel as nib
from nibabel import processing

nii_list = glob.glob("./*MRI.nii.gz")
nii_list.sort()

for nii_name in nii_list:
    print("-----------------------------------------------")
    nii_file = nib.load(nii_name)
    save_file = nib.Nifti1Image(nii_file.get_fdata(), affine=nii_file.affine, header=nii_file.header)
    smoothed_file = processing.smooth_image(save_file, fwhm=3, mode='nearest')
    save_name = nii_name[:-7]+"_F3.nii.gz"
    nib.save(smoothed_file, save_name)
    print(nii_name)