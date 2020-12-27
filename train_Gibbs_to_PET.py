from __future__ import print_function

import os
import glob
from time import time
from matplotlib import pyplot as plt
import numpy as np

import tensorflow
from tensorflow.keras.callbacks import History, ModelCheckpoint, TensorBoard, LambdaCallback
from tensorflow.keras.losses import mean_absolute_error, mean_squared_error
from tensorflow.keras.optimizers import Adam

import Unet
import NiftiGenerator

img_rows = 256 # image is resampled to this size
img_cols = 256 # image is resampled to this size
x_data_folder = 'HURLEY_GIBBS'
y_data_folder = 'HURLEY_F3'
tag = "_HURLEY_deep4_filter64_xGIBBS_yF3"
weightfile_name = 'weights'+tag+'.h5'
model_name = 'model'+tag+'.json'
jpgprogressfile_name = 'progress'+tag
batch_size = 8 # should be smallish. 1-10
num_epochs = 25 # should train for at least 100-200 in total
steps_per_epoch = 20*89 # should be enough to be equal to one whole pass through the dataset
initial_epoch = 0 # for resuming training
load_weights = False # load trained weights for resuming training

#######################

def train():
    # set fixed random seed for repeatability
    np.random.seed(813)

    print('-'*50)
    print('Creating and compiling model...')
    print('-'*50)
    model = Unet.UNetContinuous((img_rows,img_cols,1),start_ch=64,depth=4)
    model.compile(optimizer=Adam(lr=1e-4), loss=mean_squared_error, metrics=[mean_squared_error,mean_absolute_error])
    model.summary()

    # Save the model architecture
    with open(model_name, 'w') as f:
        f.write(model.to_json())

    # optionally load weights
    if load_weights:
        model.load_weights(weightfile_name)


    print('-'*50)
    print('Setting up NiftiGenerator')
    print('-'*50)
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
    niftiGen.initialize( y_data_folder, x_data_folder, niftiGen_augment_opts, niftiGen_norm_opts )
    generator = niftiGen.generate(batch_size=batch_size)
    # get one sample for progress images
    test_x = np.load('test_x.npy')
    test_y = np.load('test_y.npy')

    print('-'*50)
    print('Preparing callbacks...')
    print('-'*50)
    history = History()
    model_checkpoint = ModelCheckpoint(weightfile_name, monitor='loss', save_best_only=True)
    tensorboard = TensorBoard(log_dir=os.path.join('tblogs','{}'.format(time())))
    display_progress = LambdaCallback( on_epoch_end= lambda epoch, logs: progresscallback_img2img(epoch, logs, model, history, fig, test_x, test_y) )


    print('-'*50)
    print('Fitting network...')
    print('-'*50)
    fig = plt.figure(figsize=(15,5))
    fig.show(False)
    model.fit( generator, steps_per_epoch=steps_per_epoch, epochs=num_epochs, initial_epoch=initial_epoch, callbacks=[history,model_checkpoint,display_progress] )

# Function to display the target and prediction
def progresscallback_img2img(epoch, logs, model, history, fig, input_x, target_y):

    fig.clf()
    a = fig.add_subplot(1, 4, 1)
    plt.imshow(np.rot90(np.squeeze(input_x)),cmap='gray')
    a.axis('off')
    a.set_title('input X[0]')
    a = fig.add_subplot(1, 4, 2)
    plt.imshow(np.rot90(np.squeeze(target_y)),cmap='gray')
    a.axis('off')
    a.set_title('target Y[0]')
    a = fig.add_subplot(1, 4, 3)
    pred_y = model.predict(input_x)
    plt.imshow(np.rot90(np.squeeze(pred_y)),cmap='gray')
    a.axis('off')
    a.set_title('pred. at ' + repr(epoch+1))
    a = fig.add_subplot(1, 4, 4)
    plt.plot(range(epoch+1),history.history['loss'],'b',label='training loss')
    #plt.plot(range(epoch+1),history.history['val_loss'],'r',label='validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    a.set_title('Losses')
    fig.tight_layout()
    fig.canvas.draw()
    fig.savefig('progress_image_{0}_{1:05d}.jpg'.format(jpgprogressfile_name,epoch+1))
    fig.canvas.flush_events()

if __name__ == '__main__':
    train()
