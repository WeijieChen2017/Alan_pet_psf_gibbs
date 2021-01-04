from utils import NiftiGenerator

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
    "x_data_folder" : 'data_X',
    "y_data_folder" : 'data_Y',
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
niftiGen.initialize("./data_train/"+train_para["y_data_folder"],
                    "./data_train/"+train_para["x_data_folder"],
                    niftiGen_augment_opts, niftiGen_norm_opts )
generator = niftiGen.generate(Xslice_samples=train_para["channel_X"],
                              Yslice_samples=train_para["channel_Y"],
                              batch_size=train_para["batch_size"])