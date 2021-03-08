import glob
import os
import numpy as np

dataX_list = glob.glob("../data_train/unsplash/imgX_*.npy")
dataX_list.sort()
for pair_path_X in dataX_list:
    pair_path_Y = pair_path_X.replace("X", "Y")
    print(pair_path_X, pair_path_Y)

    data_X = np.load(pair_path_X)
    data_Y = np.load(pair_path_Y)

    n_samples = data_X.shape[2]
    norm_X = np.zeros((n_samples, data_X.shape[0], data_X.shape[1], 1), dtype=np.single)
    norm_Y = np.zeros((n_samples, data_Y.shape[0], data_Y.shape[1], 1), dtype=np.single)
    
    for idx in range(n_samples):
        norm_X[idx, :, :, :] = np.expand_dims(data_X[:, :, idx], axis=-1)
        norm_Y[idx, :, :, :] = np.expand_dims(data_Y[:, :, idx], axis=-1)

    norm_X = (norm_X-np.amin(norm_X))/np.amax(norm_X)
    norm_Y = (norm_Y-np.amin(norm_Y))/np.amax(norm_Y)
    print(np.std(norm_X), np.std(norm_Y))

    name_X = "../data_train/unsplash/norm_" + os.path.basename(pair_path_X)
    name_Y = "../data_train/unsplash/norm_" + os.path.basename(pair_path_Y)

    print(norm_X.shape, np.amax(norm_X), np.amin(norm_X))
    print(norm_Y.shape, np.amax(norm_Y), np.amin(norm_Y))    

    np.save(name_X, norm_X)
    np.save(name_Y, norm_Y)
     