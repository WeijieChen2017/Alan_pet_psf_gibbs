from __future__ import print_function

import os
import glob
import json
import random
from time import time
from matplotlib import pyplot as plt
import numpy as np

import tensorflow
from tensorflow.keras.callbacks import History, ModelCheckpoint, TensorBoard, LambdaCallback
from tensorflow.keras.losses import mean_absolute_error, mean_squared_error
from tensorflow.keras.optimizers import Adam

from models import Unet
from utils import NiftiGenerator

para_name = "us101"
# Data to be written  
train_para ={  
    "para_name" : para_name,
    "img_rows" : 512, # image is resampled to this size
    "img_cols" : 512, # image is resampled to this size
    "channel_X" : 1,
    "channel_Y" : 1,
    "start_ch" : 64,
    "depth" : 4, 
    "validation_split" : 0.2,
    "loss" : "l2",
    "data_folder" : 'unsplash',
    "data_prefix_X" : "norm_imgX",
    "data_prefix_Y" : "norm_imgY",
    "weightfile_name" : 'weights_'+para_name+'.h5',
    "model_name" : 'model_'+para_name+'.json',
    "save_folder" : './achives/',
    "batch_size" : 4, # should be smallish. 1-10
    "num_epochs" : 1, # should train for at least 100-200 in total
    "steps_per_epoch" : 8, # should be enough to be equal to one whole pass through the dataset
    "initial_epoch" : 0, # for resuming training
    "load_weights" : False, # load trained weights for resuming training
}  
     
with open("./json/train_para_"+train_para["para_name"]+".json", "w") as outfile:  
    json.dump(train_para, outfile) 

#######################

def train():
    # set fixed random seed for repeatability

    np.random.seed(813)
    if train_para["loss"] == "l1":
        loss_fn = mean_absolute_error
    if train_para["loss"] == "l2":
        loss_fn = mean_squared_error

    print(train_para)

    list_t, list_v = split_dataset_simple(data_prefix_X=train_para["data_prefix_X"],
                                          data_prefix_Y=train_para["data_prefix_Y"],
                                          data_folder="./data_train/"+train_para["data_folder"]+"/", 
                                          validation_ratio=train_para["validation_split"])
    print("Training:", list_t)
    print("Validation:", list_v)

    print('-'*50)
    print('Creating and compiling model...')
    print('-'*50)
    model = Unet.UNetContinuous(img_shape=(train_para["img_rows"],
                                           train_para["img_cols"],
                                           train_para["channel_X"]),
                                 out_ch=train_para["channel_Y"],
                                 start_ch=train_para["start_ch"],
                                 depth=train_para["depth"])
    model.compile(optimizer=Adam(lr=1e-4), loss=loss_fn,
                  metrics=[mean_squared_error,mean_absolute_error])
    model.summary()

    # Save the model architecture
    with open(train_para["save_folder"]+train_para["model_name"], 'w') as f:
        f.write(model.to_json())

    # optionally load weights
    if train_para["load_weights"]:
        model.load_weights(train_para["save_folder"]+train_para["weightfile_name"])  

    # print('-'*50)
    # print('Preparing callbacks...')
    # print('-'*50)
    # history = History()
    model_checkpoint = ModelCheckpoint(train_para["save_folder"]+train_para["weightfile_name"],
                                       monitor='loss', 
                                       save_best_only=True)
    tensorboard = TensorBoard(log_dir=os.path.join('tblogs','{}'.format(time())))
    # display_progress = LambdaCallback(on_epoch_end= lambda epoch,
    #                                   logs: progresscallback_img2img_multiple(epoch, logs, model, history, fig, generatorV) )

    print('-'*50)
    print('Fitting network...')
    print('-'*50)

    for idx_s in range(train_para["steps_per_epoch"]):
        print("Steps: ", idx_s+1)
        print('-'*20)
        print("--Training:")

        for data_pair in list_t:
            path_X = data_pair[0]
            path_Y = data_pair[1]

            try:
                data_X = np.load(path_X)
                data_Y = np.load(path_Y)
            except:
                print(path_X, path_Y)
            else:
                model.fit(x=data_X, y=data_Y, batch_size=train_para["batch_size"], epochs=train_para["num_epochs"], callbacks=[model_checkpoint])
            
        print('-'*20)
        print("--Validation:")

        for idx_p, data_pair in enumerate(list_v):
            path_X = data_pair[0]
            path_Y = data_pair[1]

            try:
                data_X = np.load(path_X)
                data_Y = np.load(path_Y)
            except:
                print(path_X, path_Y)
            else:
                model.evaluate(x=data_X, y=data_Y, batch_size=train_para["batch_size"])

                batch_X = data_X[:train_para["batch_size"], :, :, :]
                batch_Y = data_Y[:train_para["batch_size"], :, :, :]
                predictions = model.predict(batch_X)

                for idx_b in range(train_para["batch_size"]):
                    plt.figure(figsize=(8, 4), dpi=300)
                    plt.subplot(2, 3, 1)
                    plt.imshow(np.rot90(np.squeeze(batch_X[idx_b, :, :, :])),cmap='gray')
                    plt.axis('off')
                    plt.title('input X[0]')

                    plt.subplot(2, 3, 2)
                    plt.imshow(np.rot90(np.squeeze(batch_Y[idx_b, :, :, :])),cmap='gray')
                    plt.axis('off')
                    plt.title('target Y[0]')

                    plt.subplot(2, 3, 3)
                    plt.imshow(np.rot90(np.squeeze(predictions[idx_b, :, :, :])),cmap='gray')
                    plt.axis('off')
                    plt.title('pred')

                    plt.savefig('U_s{0:02d}_p{1:02d}_b{2:02d}.jpg'.format(idx_s+1, idx_p+1, idx_b+1))
                    plt.close('all')
            
            # loss_v[idx_epochs*train_para["steps_per_epoch"]+idx_steps] = loss

        model.save_weights(train_para["save_folder"]+train_para["weightfile_name"], save_format="h5")
        model.save(train_para["save_folder"]+train_para["weightfile_name"][:-3])
        # np.save(train_para["save_folder"]+train_para["weightfile_name"][:-3]+"_loss_t.npy", loss_t)
        # np.save(train_para["save_folder"]+train_para["weightfile_name"][:-3]+"_loss_v.npy", loss_v)
        # print("Checkpoints saved for epochs ", idx_epochs+1)

    # print('-'*50)
    # print('Fitting network...')
    # print('-'*50)
    # loss_fn = loss
    # optimizer = Adam(lr=1e-4)
    # loss_t = np.zeros((train_para["steps_per_epoch"]*train_para["num_epochs"]))
    # loss_v = np.zeros((train_para["steps_per_epoch"]*train_para["num_epochs"]))
    # n_train = len(list_t)
    # model.compile(optimizer=optimizer,loss=loss_fn, metrics=[mean_squared_error,mean_absolute_error])

    # for idx_epochs in range(train_para["num_epochs"]):

    #     print('-'*50)
    #     print("Epochs: ", idx_epochs+1)
    #     print('-'*20)
    #     random.shuffle(list_t)
    #     for idx_steps in range(train_para["steps_per_epoch"]):
    #         print("Steps: ", idx_steps+1)
    #         print('-'*20)
    #         print("--Training:")
    #         step_loss = 0
    #         for data_pair in list_t:
    #             path_X = data_pair[0]
    #             path_Y = data_pair[1]

    #             data_X = np.load(path_X)
    #             data_Y = np.load(path_Y)

    #             # 512, 512, 1000
    #             len_batch = train_para["batch_size"]
    #             batch_X = np.zeros((len_batch, train_para["img_rows"], train_para["img_cols"], train_para["channel_X"]))
    #             batch_Y = np.zeros((len_batch, train_para["img_rows"], train_para["img_cols"], train_para["channel_Y"]))
    #             n_iter = data_X.shape[2] // len_batch
    #             batch_loss = 0
    #             for idx_batch in range(n_iter):
    #                 batch_X[:, :, :, :] = data_X[idx_batch*len_batch:(idx_batch+1)*len_batch, :, :, :]
    #                 batch_Y[:, :, :, :] = data_Y[idx_batch*len_batch:(idx_batch+1)*len_batch, :, :, :]

    #                 # Open a GradientTape.
    #                 with tensorflow.GradientTape() as tape:
    #                     # Forward pass.
    #                     predictions = model([batch_X])
    #                     # Compute the loss value for this batch.
    #                     loss_value = loss_fn(batch_Y, predictions)
    #                     batch_loss += loss_value
    #                     # print(loss_idx_mri)
    #                     print("Training loss: ", np.mean(loss_value))

    #                 # Get gradients of loss wrt the *trainable* weights.
    #                 gradients = tape.gradient(loss_value, model.trainable_weights)
    #                 # Update the weights of the model.
    #                 optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    #             step_loss += batch_loss / n_iter
    #             print("----", os.path.basename(path_X), step_loss)
                
    #         loss_t[idx_epochs*train_para["steps_per_epoch"]+idx_steps] += step_loss / n_train
            
    #         print('-'*20)
    #         print("--Validation:")
    #         step_loss = 0
    #         for idx_p, data_pair in enumerate(list_v):
    #             path_X = data_pair[0]
    #             path_Y = data_pair[1]

    #             data_X = np.load(path_X)
    #             data_Y = np.load(path_Y)

    #             # 512, 512, 1000
    #             len_batch = train_para["batch_size"]
    #             batch_X = np.zeros((len_batch, train_para["img_rows"], train_para["img_cols"], train_para["channel_X"]))
    #             batch_Y = np.zeros((len_batch, train_para["img_rows"], train_para["img_cols"], train_para["channel_Y"]))
    #             n_iter = data_X.shape[2] // len_batch
    #             batch_loss = 0
    #             for idx_batch in range(n_iter):
    #                 batch_X[:, :, :, :] = data_X[idx_batch*len_batch:(idx_batch+1)*len_batch, :, :, :]
    #                 batch_Y[:, :, :, :] = data_Y[idx_batch*len_batch:(idx_batch+1)*len_batch, :, :, :]

    #                 predictions = model.predict(batch_X)
    #                 loss_value = loss_fn(batch_Y, predictions)
    #                 batch_loss += loss_value

    #             step_loss += batch_loss / n_iter

    #             for idx_b in range(len_batch):
    #                 plt.figure(figsize=(20, 6), dpi=300)
    #                 plt.subplot(2, 3, 1)
    #                 plt.imshow(np.rot90(np.squeeze(batch_X[idx_b, :, :, :])),cmap='gray')
    #                 plt.axis('off')
    #                 plt.title('input X[0]')

    #                 plt.subplot(2, 3, 2)
    #                 plt.imshow(np.rot90(np.squeeze(batch_Y[idx_b, :, :, :])),cmap='gray')
    #                 plt.axis('off')
    #                 plt.title('target Y[0]')

    #                 plt.subplot(2, 3, 3)
    #                 plt.imshow(np.rot90(np.squeeze(predictions[idx_b, :, :, :])),cmap='gray')
    #                 plt.axis('off')
    #                 plt.title('pred')

    #                 plt.savefig('U_e{1:02d}_s{1:02d}_p{1:02d}_b{1:02d}.jpg'.format(epoch+1, idx_steps+1, idx_p+1, idx_b+1))
            
    #         loss_v[idx_epochs*train_para["steps_per_epoch"]+idx_steps] += step_loss / n_train

    #     model.save_weights(train_para["save_folder"]+train_para["weightfile_name"], save_format="h5")
    #     model.save(train_para["save_folder"]+train_para["weightfile_name"][:-3])
    #     np.save(train_para["save_folder"]+train_para["weightfile_name"][:-3]+"_loss_t.npy", loss_t)
    #     np.save(train_para["save_folder"]+train_para["weightfile_name"][:-3]+"_loss_v.npy", loss_v)
    #     print("Checkpoints saved for epochs ", idx_epochs+1)

    # model.save_weights(train_para["save_folder"]+train_para["weightfile_name"], save_format="h5")
    # model.save(train_para["save_folder"]+train_para["weightfile_name"][:-3])


def split_dataset_simple(data_prefix_X, data_prefix_Y, data_folder, validation_ratio):

    dataX_list = glob.glob(data_folder+data_prefix_X+"*.npy")
    dataY_list = glob.glob(data_folder+data_prefix_Y+"*.npy")
    cnt_t = 0
    cnt_v = 0
    cnt_v_max = int(len(dataX_list)*validation_ratio)
    cnt_t_max = len(dataX_list) - cnt_v_max
    list_t = []
    list_v = []
    for pair_path_X in dataX_list:
        pair_name_X = os.path.basename(pair_path_X)
        pair_name_Y = pair_name_X.replace("X", "Y")
        print(pair_name_X, pair_name_Y)

        # exchange X and Y
        if cnt_t < cnt_t_max:
            cnt_t += 1
            list_t.append([data_folder+pair_name_Y, data_folder+pair_name_X])
        else:
            if cnt_v < cnt_v_max:
                cnt_v += 1
                list_v.append([data_folder+pair_name_Y, data_folder+pair_name_X])
            else:
                print("Error in dataset division.")

    return list_t, list_v

def progresscallback_img2img_multiple(epoch, logs, model, history, fig, generatorV):

    fig.clf()

    for data in generatorV:
        dataX, dataY = data
        print(dataX.shape, dataY.shape)
        sliceX = dataX.shape[3]
        sliceY = dataY.shape[3]
        break

    predY = model.predict(dataX)

    for idx in range(8):

        plt.figure(figsize=(20, 6), dpi=300)
        plt.subplot(1, 3, 1)
        plt.imshow(np.rot90(np.squeeze(dataX[idx, :, :, sliceX//2])),cmap='gray')
        plt.axis('off')
        plt.title('input X[0]')

        plt.subplot(1, 3, 2)
        plt.imshow(np.rot90(np.squeeze(dataY[idx, :, :, sliceY//2])),cmap='gray')
        plt.axis('off')
        plt.title('target Y[0]')

        plt.subplot(1, 3, 3)
        plt.imshow(np.rot90(np.squeeze(predY[idx, :, :, sliceY//2])),cmap='gray')
        plt.axis('off')
        plt.title('pred. at ' + repr(epoch+1))

        plt.savefig('progress_image_{0}_{1:05d}_samples_{1:02d}.jpg'.format(train_para["jpgprogressfile_name"], epoch+1, idx+1))

    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(range(epoch+1),history.history['loss'],'b',label='training loss')
    plt.plot(range(epoch+1),history.history['val_loss'],'r',label='validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.yscale('log')
    plt.legend()
    plt.title('Losses')
    fig.tight_layout()
    plt.savefig('progress_image_{0}_{1:05d}_loss.jpg'.format(train_para["jpgprogressfile_name"], epoch+1))

if __name__ == '__main__':
    train()
