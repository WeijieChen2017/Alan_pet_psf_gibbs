from __future__ import print_function

import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import glob
from time import time
import nibabel
import numpy as np
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from keras.losses import mean_absolute_error, mean_squared_error
from keras import backend as K

import dataUtilities as du

img_rows = 256 # image is resampled to this size 
img_cols = 256 # image is resampled to this size
img_slcs = 182 # should be fixed for all inputs
train_volumes = 60 # number of volumes used in training
data_folder = 'C:\\Users\\axm3\\Documents\\data\\deepTopup\\data\\eval'
weightfile_name = 'weights_bm_082218.h5'
model_name = 'model_bm_082218.json'

def eval():
    print('-'*50)
    print('Loading model...')
    print('-'*50)
    # load model structure and weightss
    with open(model_name, 'r') as f:
        model = model_from_json(f.read())
    model.load_weights(weightfile_name)

    print('-'*50)
    print('Loading images from {}...'.format(data_folder))
    print('-'*50)
    img1_files = sorted( glob.glob(os.path.join(data_folder,'MR_*.nii.gz')) )
    img2_files = sorted( glob.glob(os.path.join(data_folder,'CT_*.nii.gz')) )

    if len(img1_files) != len(img2_files):
        raise ValueError('Error: Number of MR and CT files do not match!')

    img_count = 0
    for i in range(train_volumes,len(img1_files)):
        img_count += 1

        print(' {} ===> {}'.format(img1_files[i],img2_files[i]))

        # read in MR image
        curr_img1_nii = nibabel.load(img1_files[i])
        # read in CT image
        curr_img2_nii = nibabel.load(img2_files[i])

        # get the data
        curr_img1 = np.float32( curr_img1_nii.get_data() )
        curr_img2 = np.float32( curr_img2_nii.get_data() )

        # normalize input data to mean 0 and stdev 1
        mean_img = np.mean( curr_img1 )
        std_img = np.std( curr_img1 )
        curr_img1 -= mean_img
        curr_img1 /= std_img

        # normalize CT data
        curr_img2 /= 2500

        # reshape for prediction
        samp_img1 = np.transpose( curr_img1, (2,0,1) )
        samp_img1 = du.get25DImage( samp_img1, 3 )  
        samp_img2 = np.transpose( curr_img2, (2,0,1) )

        # predict
        pred_img2 = model.predict(samp_img1, verbose=1)

        # reverse scaling
        pred_img2 *= 2500

        # reshape for save as nii
        pred_img2 = np.transpose( pred_img2, (1,2,0,3))

        # save output
        pred_img2_nii = nibabel.Nifti1Image(pred_img2, curr_img1_nii.affine, curr_img1_nii.header)
        subj_id_suffix = img1_files[i].split('.nii.gz')
        pred_img2_file = os.path.join( os.path.dirname(subj_id_suffix[0]), '..', os.path.basename(subj_id_suffix[0]) + '_SYN_CT.nii.gz' )
        nibabel.save(pred_img2_nii, pred_img2_file)



if __name__ == '__main__':
    eval()