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

from utils import dataUtilities as du
from utils import NiftiGenerator

train_para_name_hub = ["ex01"]
test_para_name_prefix = "ex"
test_count = 0


def create_index(dataA, n_slice, zeroPadding=False):
    h, w, z = dataA.shape
    index = np.zeros((z,n_slice))
    
    for idx_z in range(z):
        for idx_c in range(n_slice):
            index[idx_z, idx_c] = idx_z-(n_slice-idx_c+1)+n_slice//2+2
    if zeroPadding:
        index[index<0]=z
        index[index>z-1]=z
    else:
        index[index<0]=0
        index[index>z-1]=z-1
    return index

def createInput(data, n_slice=1):
    h, w, z = data.shape
    data_input = np.zeros((z, h, w, n_slice))
    index = create_index(data, n_slice, zeroPadding=False)
        
    for idx_z in range(z):
        for idx_c in range(n_slice):
            data_input[idx_z, :, :, idx_c] = data[:, :, int(index[idx_z, idx_c])]
            
    return data_input


def eval():
    for train_para_name in train_para_name_hub:
        test_count += 1
        test_para_name = test_para_name_prefix + "{0:0>3}".format(test_count)

        with open("./json/train_para_"+train_para_name+".json") as f:
          train_para = json.load(f)

        test_para ={  
            "test_para_name" : test_para_name,
            "train_para_name" : train_para_name,
            "channel_X" : train_para_name["channel_X"],
            "channel_Y" : train_para_name["channel_Y"], 
            "data_folder" : 'PET_RSZP_10',
        }

        print("Model: ./Achives/model_"+test_para["train_para_name"]+".json")
        model_list = glob.glob("./Achives/model_"+test_para["train_para_name"]+".json")
        model_list.sort()
        for model_path in model_list:
            weights_path = "./Achives/model_"+test_para["train_para_name"]+".json"
            print('-'*50)
            print('Loading model...')
            print('-'*50)
            print("Model path: ", model_path)
            print("Weights path: ", weights_path)
        
            with open(model_path, 'r') as f:
                model = model_from_json(f.read())
            model.load_weights(weights_path)

            print('-'*50)
            print('Loading images from {}...'.format(data_folder))
            print('-'*50)
            
            testX_list = glob.glob("./test_data/"+test_para["data_folder"]+"/*.nii")
            testX_list.sort()
            for testX_path in testX_list:
                print("testX: ", testX_path)
                testX_name = os.path.basename(testX_path)
                testX_file = nibabel.load(testX_path)
                testX_data = testX_file.get_fdata()
                testX_max = np.amax(testX_data)
                testX_norm = testX_data / testX_max
                
                # inputX = np.transpose(testX_norm, (2,0,1))
                inputX = createInput(testX_norm, n_slice=test_para["channel_X"])
                print("inputX shape: ", inputX.shape)
                outputY =  model.predict(inputX, verbose=1)
                print("outputY shape: ", np.transpose(outputY, (1,2,0,3)).shape)
                predY_data = np.squeeze(np.transpose(outputY, (1,2,0,3))[:, :, :, test_para["channel_Y"] // 2]) * testX_max
                testX_sum = np.sum(testX_data)
                predY_sum = np.sum(predY_data)
                predY_data = predY_data / predY_sum * testX_sum
                diffY_data = np.subtract(testX_data, predY_data)

                predY_folder = "./results/"+test_para["test_para_name"]+"/predY/"
                diffY_folder = "./results/"+test_para["test_para_name"]+"/diffY/"
                if not os.path.exists(predY_folder):
                    os.makedirs(predY_folder)
                if not os.path.exists(diffY_folder):
                    os.makedirs(diffY_folder)

                predY_file = nibabel.Nifti1Image(predY_data, testX_file.affine, testX_file.header)
                diffY_file = nibabel.Nifti1Image(diffY_data, testX_file.affine, testX_file.header)
                predY_name = predY_folder+testX_name
                diffY_name = diffY_folder+testX_name
                nibabel.save(predY_file, predY_name)
                nibabel.save(diffY_file, diffY_name)
        with open("./results/"+test_para["test_para_name"]+"/test_para_"+para_name+".json", "w") as outfile:  
            json.dump(test_para, outfile)
        with open("./json/"+test_para["test_para_name"]+"/test_para_"+para_name+".json", "w") as outfile:  
            json.dump(test_para, outfile)
        with open("./results/"+test_para["train_para_name"]+"/train_para_"+para_name+".json", "w") as outfile:  
            json.dump(train_para, outfile) 

if __name__ == '__main__':
    eval()