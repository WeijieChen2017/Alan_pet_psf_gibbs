from scipy.ndimage import zoom
from scipy.io import loadmat
import numpy as np
import nibabel as nib
import glob
import os

def process_data(data):
    
    px, py, pz = data.shape
    qx, qy, qz = (256, 256, 89)
    zoom_data = zoom(data, (qx/px, qy/py, qz/pz))
    
    (values,counts) = np.unique(zoom_data,return_counts=True)
    ind = np.argmax(counts)
    th_min = values[ind]
    print("background: ", th_min)
    zoom_data[zoom_data<th_min] = 0
    th_max = np.percentile(zoom_data, q=99.9)
    zoom_data[zoom_data>th_max] = th_max
    zoom_data = zoom_data / th_max
    print("Max: ", np.amax(zoom_data))
    print("Min: ", np.amin(zoom_data))


    print("Old dim:", data.shape)
    print("New dim:", zoom_data.shape)

    return zoom_data

nii_list = glob.glob("./inv_F3/*.nii")
niii_list.sort()
for nii_name in nii_list:
    print("-----------------------------------------------")
    nii_idx = os.path.basename(nii_name)[:-4]
    mat_list = glob.glob("./inv_DR/"+nii_idx+"*.mat")
    mat_name = mat_list[0]
    print(nii_name)
    print(mat_name)


    # mdict = loadmat(mat_name)

    # try:
    #     mat_data = mdict["reconImg"]
    # except Exception:
    #     pass  # or you could use 'continue'

    # try:
    #     mat_data = mdict["data"]
    # except Exception:
    #     pass  # or you could use 'continue'

    # save_data = process_data(mat_data)
    # save_file = nib.Nifti1Image(save_data, affine=tmpl_affine, header=tmpl_header)
    # save_name = "./inv_GIBBS"+nii_idx+".nii"
    # nib.save(save_file, save_name)
    # print(save_name)
