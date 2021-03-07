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
    "data_prefix_X" : "imgX",
    "data_prefix_Y" : "imgY",
    "weightfile_name" : 'weights_'+para_name+'.h5',
    "model_name" : 'model_'+para_name+'.json',
    "save_folder" : './achives/',
    "jpgprogressfile_name" : 'progress_'+para_name,
    "batch_size" : 8, # should be smallish. 1-10
    "num_epochs" : 25, # should train for at least 100-200 in total
    "steps_per_epoch" : 10, # should be enough to be equal to one whole pass through the dataset
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
        loss = mean_absolute_error
    if train_para["loss"] == "l2":
        loss = mean_squared_error

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
    model.compile(optimizer=Adam(lr=1e-4),
                  loss=loss,
                  metrics=[mean_squared_error,mean_absolute_error])
    model.summary()

    # Save the model architecture
    with open(train_para["save_folder"]+train_para["model_name"], 'w') as f:
        f.write(model.to_json())

    # optionally load weights
    if train_para["load_weights"]:
        model.load_weights(train_para["save_folder"]+train_para["weightfile_name"])  

    print('-'*50)
    print('Preparing callbacks...')
    print('-'*50)
    history = History()
    model_checkpoint = ModelCheckpoint(train_para["save_folder"]+train_para["weightfile_name"],
                                       monitor='val_loss', 
                                       save_best_only=True)
    tensorboard = TensorBoard(log_dir=os.path.join('tblogs','{}'.format(time())))
    display_progress = LambdaCallback(on_epoch_end= lambda epoch,
                                      logs: progresscallback_img2img_multiple(epoch, logs, model, history, fig, generatorV) )

    print('-'*50)
    print('Fitting network...')
    print('-'*50)
    loss_fn = loss
    optimizer = Adam(lr=1e-4)
    loss_t = np.zeros((train_para["steps_per_epoch"]*train_para["num_epochs"]))
    loss_v = np.zeros((train_para["num_epochs"]))
    n_train = len(list_t)
    model.compile(optimizer=optimizer,loss=loss_fn, metrics=[mean_squared_error,mean_absolute_error])

    for idx_epochs in range(train_para["num_epochs"]):

        print('-'*50)
        print("Epochs: ", idx_epochs+1)
        print('-'*20)
        random.shuffle(list_t)
        for idx_steps in range(train_para["steps_per_epoch"]):
            print("Steps: ", idx_steps+1)
            print('-'*20)

            for data_pair in list_t:
                path_X = data_pair[0]
                path_Y = data_pair[1]

                data_X = np.load(path_X)
                data_Y = np.load(path_Y)

                print(data_X.shape)
                print(data_Y.shape)
            exit()

            

        for batch_X, batch_Y, batch_Z in generatorT:

            print("#"*6, idx_eM, "MRI Phase:")
            print(np.mean(batch_X))
            print(np.mean(batch_Y))
            print(np.mean(batch_Z))

            # Open a GradientTape.
            with tensorflow.GradientTape() as tape:
                # Forward pass.
                predictions = model([batch_X, batch_Z, 
                                     np.ones((1, )), np.zeros((1, ))])
                # Compute the loss value for this batch.
                loss_value = loss_fn(batch_Y, predictions)
                loss_idx_mri = idx_epochs*train_para["epoch_per_MRI"]+idx_eM
                # print(loss_idx_mri)
                loss_mri[loss_idx_mri] = np.mean(loss_value)
                print("Phase MRI loss: ", np.mean(loss_value))

            # Get gradients of loss wrt the *trainable* weights.
            gradients = tape.gradient(loss_value, model.trainable_weights)
            # Update the weights of the model.
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))
            if idx_eM >= train_para["epoch_per_MRI"]:
                break
            else:
                idx_eM += 1

        # train PET
        idx_eP = 1
        model = freeze_phase(model, phase="PET")
        model.compile(optimizer=optimizer,loss=loss_fn, metrics=[mean_squared_error,mean_absolute_error])

        # Iterate over the batches of a dataset.
        for batch_X, batch_Y, batch_Z in generatorT:

            print("@"*6, idx_eP, "PET Phase:")
            print(np.mean(batch_X))
            print(np.mean(batch_Y))
            print(np.mean(batch_Z))

            # Open a GradientTape.
            with tensorflow.GradientTape() as tape:
                # Forward pass.
                predictions = model([batch_X, batch_Z,
                                     np.zeros((1, )), np.ones((1, ))])
                # Compute the loss value for this batch.
                gt_Z = np.expand_dims(batch_Z[:, :, :, train_para["channel_Z"]//2], axis=3)
                loss_value = loss_fn(gt_Z, predictions)
                loss_idx_pet = idx_epochs*train_para["epoch_per_MRI"]+idx_eP
                # print(loss_idx_pet)
                loss_pet[loss_idx_pet] = np.mean(loss_value)
                print("Phase PET loss: ", np.mean(loss_value))

            
            # Get gradients of loss wrt the *trainable* weights.
            gradients = tape.gradient(loss_value, model.trainable_weights)
            # Update the weights of the model.
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))
            if idx_eP >= train_para["epoch_per_PET"]:
                break
            else:
                idx_eP += 1

        if idx_epochs % train_para["save_per_epochs"] == 0:
            model.save_weights(train_para["save_folder"]+train_para["weightfile_name"], save_format="h5")
            model.save(train_para["save_folder"]+train_para["weightfile_name"][:-3])
            np.save(train_para["save_folder"]+train_para["weightfile_name"][:-3]+"_loss_mri.npy", loss_mri)
            np.save(train_para["save_folder"]+train_para["weightfile_name"][:-3]+"_loss_pet.npy", loss_pet)
            print("Checkpoints saved for epochs ", idx_epochs+1)
        if idx_epochs % train_para["eval_per_epochs"] == 0:
            print("Save eval images.")
            progress_eval(generator=generatorV, model=model, loss_fn=loss_fn,
                          epochs=idx_epochs+1, img_num = train_para["eval_num_img"],
                          save_name = train_para["jpgprogressfile_name"])
        if idx_epochs >= train_para["steps_per_epoch"] * train_para["num_epochs"] + 1:
            break

    model.save_weights(train_para["save_folder"]+train_para["weightfile_name"], save_format="h5")
    model.save(train_para["save_folder"]+train_para["weightfile_name"][:-3])
    os.system("mkdir "+train_para["para_name"])
    os.system("mv *"+train_para["para_name"]+"*.jpg "+train_para["para_name"])
    os.system("mv "+train_para["para_name"]+" ./jpeg/")


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

        if cnt_t < cnt_t_max:
            cnt_t += 1
            list_t.append([data_folder+pair_name_X, data_folder+pair_name_Y])
        else:
            if cnt_v < cnt_v_max:
                cnt_v += 1
                list_v.append([data_folder+pair_name_X, data_folder+pair_name_Y])
            else:
                print("Error in dataset division.")

    return list_t, list_v
     


def dataset_go_back(folder_list, sub_folder_list):

    [folderX, folderY] = folder_list
    [train_folderX, train_folderY, valid_folderX, valid_folderY] = sub_folder_list
    
    data_trainX_list = glob.glob(train_folderX+"/*.nii")+glob.glob(train_folderX+"/*.nii.gz")
    data_validX_list = glob.glob(valid_folderX+"/*.nii")+glob.glob(valid_folderX+"/*.nii.gz")
    data_trainY_list = glob.glob(train_folderY+"/*.nii")+glob.glob(train_folderY+"/*.nii.gz")
    data_validY_list = glob.glob(valid_folderY+"/*.nii")+glob.glob(valid_folderY+"/*.nii.gz")

    for data_path in data_trainX_list:
        cmd = "mv "+data_path+" "+folderX
        os.system(cmd)

    for data_path in data_validX_list:
        cmd = "mv "+data_path+" "+folderX
        os.system(cmd)

    for data_path in data_trainY_list:
        cmd = "mv "+data_path+" "+folderY
        os.system(cmd)

    for data_path in data_validY_list:
        cmd = "mv "+data_path+" "+folderY
        os.system(cmd)

# Split the dataset and move them to the corresponding folder
def split_dataset(folderX, folderY, validation_ratio):

    train_folderX = folderX + "/trainX/"
    train_folderY = folderY + "/trainY/"
    valid_folderX = folderX + "/validX/"
    valid_folderY = folderY + "/validY/"

    if not os.path.exists(train_folderX):
        os.makedirs(train_folderX)
    if not os.path.exists(train_folderY):
        os.makedirs(train_folderY)
    if not os.path.exists(valid_folderX):
        os.makedirs(valid_folderX)
    if not os.path.exists(valid_folderY):
        os.makedirs(valid_folderY)


    data_path_list = glob.glob(folderX+"/*.nii") + glob.glob(folderX+"/*.nii.gz")
    data_path_list.sort()
    data_path_list = np.asarray(data_path_list)
    np.random.shuffle(data_path_list)
    data_path_list = list(data_path_list)
    data_name_list = []
    for data_path in data_path_list:
        data_name_list.append(os.path.basename(data_path))

    valid_list = data_name_list[:int(len(data_name_list)*validation_ratio)]
    valid_list.sort()
    train_list = list(set(data_name_list) - set(valid_list))
    train_list.sort()

    print("valid_list: ", valid_list)
    print('-'*50)
    print("train_list: ", train_list)

    for valid_name in valid_list:
        valid_nameX = folderX+"/"+valid_name
        valid_nameY = folderY+"/"+valid_name
        cmdX = "mv "+valid_nameX+" "+valid_folderX
        cmdY = "mv "+valid_nameY+" "+valid_folderY
        # print(cmdX)
        # print(cmdY)
        os.system(cmdX)
        os.system(cmdY)

    for train_name in train_list:
        train_nameX = folderX+"/"+train_name
        train_nameY = folderY+"/"+train_name
        cmdX = "mv "+train_nameX+" "+train_folderX
        cmdY = "mv "+train_nameY+" "+train_folderY
        # print(cmdX)
        # print(cmdY)
        os.system(cmdX)
        os.system(cmdY)

    return [train_folderX, train_folderY, valid_folderX, valid_folderY]


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


# Function to display the target and prediction
def progresscallback_img2img(epoch, logs, model, history, fig, generatorV):

    fig.clf()

    for data in generatorV:
        dataX, dataY = data
        print(dataX.shape, dataY.shape)
        sliceX = dataX.shape[3]
        sliceY = dataY.shape[3]
        break

    predY = model.predict(dataX)

    for idx in range(4):
        a = fig.add_subplot(4, 4, idx+5)
        plt.imshow(np.rot90(np.squeeze(dataX[idx, :, :, sliceX//2])),cmap='gray')
        a.axis('off')
        a.set_title('input X[0]')
        a = fig.add_subplot(4, 4, idx+9)
        plt.imshow(np.rot90(np.squeeze(dataY[idx, :, :, sliceY//2])),cmap='gray')
        a.axis('off')
        a.set_title('target Y[0]')
        a = fig.add_subplot(4, 4, idx+13)
        plt.imshow(np.rot90(np.squeeze(predY[idx, :, :, sliceY//2])),cmap='gray')
        a.axis('off')
        a.set_title('pred. at ' + repr(epoch+1))

    a = fig.add_subplot(4, 1, 1)
    plt.plot(range(epoch+1),history.history['loss'],'b',label='training loss')
    plt.plot(range(epoch+1),history.history['val_loss'],'r',label='validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.yscale('log')
    plt.legend()
    a.set_title('Losses')
    fig.tight_layout()
    fig.canvas.draw()
    fig.savefig('progress_image_{0}_{1:05d}.jpg'.format(train_para["jpgprogressfile_name"],epoch+1))
    fig.canvas.flush_events()

if __name__ == '__main__':
    train()
