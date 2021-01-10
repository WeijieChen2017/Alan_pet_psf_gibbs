from utils import NiftiGenerator
from matplotlib import pyplot as plt
import numpy as np

para_name = "ex99"
# Data to be written  
train_para ={  
    "para_name" : para_name,
    "img_rows" : 256, # image is resampled to this size
    "img_cols" : 256, # image is resampled to this size
    "channel_X" : 1,
    "channel_Y" : 1,
    "start_ch" : 64,
    "depth" : 3, 
    # "validation_split" : 0.5,
    "loss" : "l2",
    "x_data_folder" : 'dataX',
    "y_data_folder" : 'dataY',
    "weightfile_name" : 'weights_'+para_name+'.h5',
    "model_name" : 'model_'+para_name+'.json',
    "save_folder" : './achives/',
    "jpgprogressfile_name" : 'progress_'+para_name,
    "batch_size" : 8, # should be smallish. 1-10
    "num_epochs" : 25, # should train for at least 100-200 in total
    "steps_per_epoch" : 20*89, # should be enough to be equal to one whole pass through the dataset
    "initial_epoch" : 0, # for resuming training
    "load_weights" : False, # load trained weights for resuming training
}  

niftiGen = NiftiGenerator.PairedNiftiGenerator()
niftiGen_augment_opts = NiftiGenerator.PairedNiftiGenerator.get_default_augOptions()
niftiGen_augment_opts.hflips = True
niftiGen_augment_opts.vflips = True
niftiGen_augment_opts.rotations = 15
niftiGen_augment_opts.scalings = 0
niftiGen_augment_opts.shears = 0
niftiGen_augment_opts.translations = 10
print(niftiGen_augment_opts)
niftiGen_norm_opts = NiftiGenerator.PairedNiftiGenerator.get_default_normOptions()
niftiGen_norm_opts.normXtype = 'auto'
niftiGen_norm_opts.normYtype = 'auto'
print(niftiGen_norm_opts)
niftiGen.initialize("./data_train/"+train_para["x_data_folder"],
                    "./data_train/"+train_para["y_data_folder"],
                    niftiGen_augment_opts, niftiGen_norm_opts )
generator = niftiGen.generate(Xslice_samples=train_para["channel_X"],
                              Yslice_samples=train_para["channel_Y"],
                              batch_size=train_para["batch_size"])

fig = plt.figure(figsize=(15,5))
fig.show(False)
cnt = 0
for idx, data in enumerate(generator):
    (dataX, dataY) = data
    print(idx)
    print(dataX.shape, dataY.shape)
    print("%"*72)
    fig.clf()
    a = fig.add_subplot(1, 2, 1)
    plt.imshow(np.rot90(np.squeeze(dataX[3, :, :, :])),cmap='gray')
    a.axis('off')
    a.set_title('dataX')
    a = fig.add_subplot(1, 2, 2)
    plt.imshow(np.rot90(np.squeeze(dataY[3, :, :, :])),cmap='gray')
    a.axis('off')
    a.set_title('dataY')

    fig.tight_layout()
    fig.canvas.draw()
    fig.savefig('progress_image_{0}_{1:05d}.jpg'.format(train_para["jpgprogressfile_name"],idx+1))
    fig.canvas.flush_events()
    cnt += 1
    if cnt > 8:
        break
