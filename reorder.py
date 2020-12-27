from scipy.ndimage import zoom
from scipy.io import loadmat
import numpy as np
import nibabel as nib
import glob
import os

name_dataset = "HURLEY"
Y_list = glob.glob("./"+name_dataset+"_F3/*.nii")
Y_list.sort()
new_order = 1

for Y_name in Y_list:
    print("-----------------------------------------------")
    Y_idx = os.path.basename(Y_name)[:-4]
    X_list = glob.glob("./"+name_dataset+"_GIBBS/"+Y_idx+"*.nii")
    X_name = X_list[0]
    print("original X:", X_name)
    print("original Y:", Y_name)

    idx_str = "{0:0>3}".format(new_order)
    cmd_X = "mv "+X_name+" h"+idx_str+".nii"
    cmd_Y = "mv "+Y_name+" h"+idx_str+".nii"
    print(cmd_X)
    print(cmd_Y)

