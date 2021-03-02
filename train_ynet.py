

from __future__ import print_function

import os
import glob
import json
from time import time
from matplotlib import pyplot as plt
import numpy as np

import tensorflow
from tensorflow.keras.callbacks import History, ModelCheckpoint, TensorBoard, LambdaCallback
from tensorflow.keras.losses import mean_absolute_error, mean_squared_error
from tensorflow.keras.optimizers import Adam

from models import Ynet
from utils import NiftiGenerator

para_name = "ynet07"
# Data to be written  
train_para ={  
    "para_name" : para_name,
    "img_rows" : 512, # image is resampled to this size
    "img_cols" : 512, # image is resampled to this size
    "channel_X" : 5,
    "channel_Y" : 1,
    "channel_Z" : 5,
    "start_ch" : 64,
    "depth" : 3,
    "epoch_per_MRI": 1,
    "epoch_per_PET": 5,
    "validation_split" : 0.2,
    "loss" : "l2",
    "x_data_folder" : 'MRI_232_F3',
    "y_data_folder" : 'MRI_232',
    "z_data_folder" : 'antspy',
    "weightfile_name" : 'weights_'+para_name+'.h5',
    "model_name" : 'model_'+para_name+'.json',
    "save_folder" : './achives/',
    "save_per_epochs" : 1000,
    "eval_per_epochs" : 400,
    "eval_num_img" : 4,
    "jpgprogressfile_name" : para_name,
    "batch_size" : 2, # should be smallish. 1-10
    "num_epochs" : 5, # should train for at least 100-200 in total
    "steps_per_epoch" : 2400, # should be enough to be equal to one whole pass through the dataset
    "initial_epoch" : 0, # for resuming training
    "load_weights" : False, # load trained weights for resuming training
}  
     
with open("./json/train_para_"+train_para["para_name"]+".json", "w") as outfile:  
    json.dump(train_para, outfile) 

#######################

def train():
    # set fixed random seed for repeatability

    print(train_para)

    np.random.seed(813)
    if train_para["loss"] == "l1":
        loss = mean_absolute_error
    if train_para["loss"] == "l2":
        loss = mean_squared_error

    print('-'*50)
    print('Creating and compiling model...')
    print('-'*50)

    model = Ynet.YNet(img_shape_PET=(train_para["img_rows"],
                                     train_para["img_cols"],
                                     train_para["channel_X"]),
                      img_shape_MRI=(train_para["img_rows"],
                                     train_para["img_cols"],
                                     train_para["channel_X"]),
                      out_ch=train_para["channel_Y"],
                      start_ch=train_para["start_ch"],
                      depth=train_para["depth"])
    model.summary()

    # Save the model architecture
    with open(train_para["save_folder"]+train_para["model_name"], 'w') as f:
        f.write(model.to_json())

    # optionally load weights
    if train_para["load_weights"]:
        model.load_weights(train_para["save_folder"]+train_para["weightfile_name"])


    print('-'*50)
    print('Setting up NiftiGenerator')
    print('-'*50)
    niftiGen_augment_opts = NiftiGenerator.TripleNiftiGenerator.get_default_augOptions()
    niftiGen_augment_opts.hflips = True
    niftiGen_augment_opts.vflips = True
    niftiGen_augment_opts.rotations = 15
    niftiGen_augment_opts.scalings = 0
    niftiGen_augment_opts.shears = 0
    niftiGen_augment_opts.translations = 10
    print(niftiGen_augment_opts)
    niftiGen_norm_opts = NiftiGenerator.TripleNiftiGenerator.get_default_normOptions()
    niftiGen_norm_opts.normXtype = 'auto'
    niftiGen_norm_opts.normYtype = 'auto'
    niftiGen_norm_opts.normZtype = 'auto'
    print(niftiGen_norm_opts)

    folderX = "./data_train/"+train_para["x_data_folder"]
    folderY = "./data_train/"+train_para["y_data_folder"]
    folderZ = "./data_train/"+train_para["z_data_folder"]
    folder_list = [folderX, folderY, folderZ]
    sub_folder_list = split_dataset_triple(folderX=folderX, folderY=folderY, folderZ = folderZ,
                                           validation_ratio=train_para["validation_split"])
    # [train_folderX, train_folderY, valid_folderX, valid_folderY] = sub_folder_list
    [train_folderX, train_folderY, train_folderZ, valid_folderX, valid_folderY, valid_folderZ] = sub_folder_list
    print(sub_folder_list)

    niftiGenT = NiftiGenerator.TripleNiftiGenerator_paired()
    niftiGenT.initialize(train_folderX, train_folderY, train_folderZ,
                         niftiGen_augment_opts, niftiGen_norm_opts)
    generatorT = niftiGenT.generate(img_size=(train_para["img_rows"],train_para["img_cols"]),
                                    Xslice_samples=train_para["channel_X"],
                                    Yslice_samples=train_para["channel_Y"],
                                    Zslice_samples=train_para["channel_Z"],
                                    batch_size=train_para["batch_size"])

    niftiGenV = NiftiGenerator.TripleNiftiGenerator_paired()
    niftiGenV.initialize(valid_folderX, valid_folderY, valid_folderZ,
                         niftiGen_augment_opts, niftiGen_norm_opts )
    generatorV = niftiGenV.generate(img_size=(train_para["img_rows"],train_para["img_cols"]),
                                    Xslice_samples=train_para["channel_X"],
                                    Yslice_samples=train_para["channel_Y"],
                                    Zslice_samples=train_para["channel_Z"],
                                    batch_size=train_para["batch_size"])

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
    loss_mri = np.zeros((train_para["steps_per_epoch"]*train_para["num_epochs"]*train_para["epoch_per_MRI"]))
    loss_pet = np.zeros((train_para["steps_per_epoch"]*train_para["num_epochs"]*train_para["epoch_per_PET"]))

    for idx_epochs in range(train_para["steps_per_epoch"] * train_para["num_epochs"] + 1):

        print('-'*50)
        print("Epochs: ", idx_epochs+1)

        # train MRI
        idx_eM = 1
        model = freeze_phase(model, phase="MRI")
        model.compile(optimizer=optimizer,loss=loss_fn, metrics=[mean_squared_error,mean_absolute_error])

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
    dataset_go_back(folder_list, sub_folder_list)
    os.system("mkdir "+train_para["para_name"])
    os.system("mv *"+train_para["para_name"]+"*.jpg "+train_para["para_name"])
    os.system("mv "+train_para["para_name"]+" ./jpeg/")


def freeze_phase(model, phase):
    PET_set = [3, 5, 9, 11, 15, 17]
    MRI_set = [2, 4, 8, 10, 14, 16, 25, 27, 29, 30, 31, 33, 34, 35, 37, 38, 39, 40]
    if phase == "MRI":
        for idx in PET_set:
            model.layers[idx].trainable = False
        for idx in MRI_set:
            model.layers[idx].trainable = True
            # print( model.layers[idx].name)

    if phase == "PET":
        for idx in PET_set:
            model.layers[idx].trainable = True
            # print( model.layers[idx].name)
        for idx in MRI_set:
            model.layers[idx].trainable = False

    return model


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
def split_dataset_triple(folderX, folderY, folderZ, validation_ratio):

    train_folderX = folderX + "/trainX/"
    train_folderY = folderY + "/trainY/"
    train_folderZ = folderZ + "/trainZ/"
    valid_folderX = folderX + "/validX/"
    valid_folderY = folderY + "/validY/"
    valid_folderZ = folderZ + "/validZ/"

    if not os.path.exists(train_folderX):
        os.makedirs(train_folderX)
    if not os.path.exists(train_folderY):
        os.makedirs(train_folderY)
    if not os.path.exists(train_folderZ):
        os.makedirs(train_folderZ)
    if not os.path.exists(valid_folderX):
        os.makedirs(valid_folderX)
    if not os.path.exists(valid_folderY):
        os.makedirs(valid_folderY)
    if not os.path.exists(valid_folderZ):
        os.makedirs(valid_folderZ)    


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
        valid_nameZ = folderZ+"/"+valid_name
        cmdX = "mv "+valid_nameX+" "+valid_folderX
        cmdY = "mv "+valid_nameY+" "+valid_folderY
        cmdZ = "mv "+valid_nameZ+" "+valid_folderZ
        # print(cmdX)
        # print(cmdY)
        os.system(cmdX)
        os.system(cmdY)
        os.system(cmdZ)

    for train_name in train_list:
        train_nameX = folderX+"/"+train_name
        train_nameY = folderY+"/"+train_name
        train_nameZ = folderZ+"/"+train_name
        cmdX = "mv "+train_nameX+" "+train_folderX
        cmdY = "mv "+train_nameY+" "+train_folderY
        cmdZ = "mv "+train_nameZ+" "+train_folderZ
        # print(cmdX)
        # print(cmdY)
        os.system(cmdX)
        os.system(cmdY)
        os.system(cmdZ)

    return [train_folderX, train_folderY, train_folderZ, valid_folderX, valid_folderY, valid_folderZ]


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


def progress_eval(generator, model, loss_fn, epochs, img_num, save_name):
    
    idx_eval = 1
    idx_gen = 1

    for batch_X, batch_Y, batch_Z in generator:
        mri_input = batch_X
        mri_output = batch_Y
        pet_input = batch_Z
        n_slice = mri_input.shape[0]

        for idx in range(n_slice):

            print("&"*6, "eval", str(idx_eval))
            mri_eval = model([mri_input, pet_input, np.ones((1, )), np.zeros((1, ))])
            mri_loss = loss_fn(mri_output, mri_eval)
            pet_eval = model([mri_input, pet_input, np.zeros((1, )), np.ones((1, ))])
            pet_gt = np.expand_dims(pet_input[:, :, :, train_para["channel_Z"]//2], axis=3)
            pet_loss = loss_fn(pet_gt, pet_eval)

            # print("mri_input", mri_input.shape)
            # print("mri_output", mri_output.shape)
            # print("mri_eval", mri_eval.shape)
            # print("pet_input", pet_input.shape)
            # print("pet_eval", pet_eval.shape)

            img_mri_input = np.squeeze(mri_input[idx, :, :, int(mri_input.shape[3]//2)])
            img_mri_output = np.squeeze(mri_output[idx, :, :, int(mri_output.shape[3]//2)])
            img_mri_eval = np.squeeze(mri_eval[idx, :, :, int(mri_eval.shape[3]//2)])
            img_pet_input = np.squeeze(pet_input[idx, :, :, int(pet_input.shape[3]//2)])
            img_pet_eval = np.squeeze(pet_eval[idx, :, :, int(pet_eval.shape[3]//2)])

            plt.figure(figsize=(12, 6), dpi=300)
            plt.subplot(2, 3, 1)
            plt.imshow(np.rot90(img_mri_input),cmap='gray')
            plt.axis('off')
            plt.title('mri_input')

            plt.subplot(2, 3, 2)
            plt.imshow(np.rot90(img_mri_eval),cmap='gray')
            plt.axis('off')
            plt.title('mri_eval')

            plt.subplot(2, 3, 3)
            plt.imshow(np.rot90(img_mri_output),cmap='gray')
            plt.axis('off')
            plt.title('mri_output')

            plt.subplot(2, 3, 4)
            plt.imshow(np.rot90(img_pet_input),cmap='gray')
            plt.axis('off')
            plt.title('pet_input')

            plt.subplot(2, 3, 5)
            plt.imshow(np.rot90(img_pet_eval),cmap='gray')
            plt.axis('off')
            plt.title('pet_eval')

            plt.title("MSR:   MRI_loss: "+str(np.mean(mri_loss))+" || PET_loss: "+str(np.mean(pet_loss)))
            plt.savefig("progress_image_{0}_e{1:06d}_samples_{2:02d}.jpg".format(save_name, epochs, idx_eval))
            print("progress_image_{0}_e{1:06d}_samples_{2:02d}.jpg".format(save_name, epochs, idx_eval))
            plt.close()

            idx_eval += 1

        if idx_gen >= img_num:
            break
        else:
            idx_gen += 1



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
