import os
import nibabel as nib

nii_list = glob.glob("./*.nii")
nii_list.sort()

for nii_name in nii_list:
    print("-----------------------------------------------")
    nii_file = nib.load(nii_name)
    save_file = nib.Nifti1Image(nii_file.get_fdata(), affine=nii_file.affine, header=nii_file.header)
    smoothed_file = processing.smooth_image(save_file, fwhm=1, mode='nearest')
    save_name = nii_name[:-4]+"_F1.nii"
    nib.save(smoothed_file, save_name)
    print(nii_name)