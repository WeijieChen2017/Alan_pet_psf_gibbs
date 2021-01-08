import nibabel as np
import nibabel as nib
from scipy.ndimage import zoom

nii_list = glob.glob("./*.nii")
nii_list.sort()

for nii_name in nii_list:
    print("-----------------------------------------------")
    nii_file = nib.load(nii_name)
    ori_data = nii_file.get_fdata()
    zoom_data = zoom(data, (2, 2, 2))

    # px, py, pz = data.shape
    # qx, qy, qz = (256, 256, 89)
    # zoom_data = zoom(data, (qx/px, qy/py, qz/pz))


    save_file = nib.Nifti1Image(zoom_data, affine=nii_file.affine, header=nii_file.header)
    # smoothed_file = processing.smooth_image(save_file, fwhm=3, mode='nearest')
    save_name = nii_name[:-4]+"_2x.nii"
    nib.save(save_file, save_name)
    print(nii_name)