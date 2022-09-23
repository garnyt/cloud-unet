# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 08:29:29 2022

Based on: https://github.com/bnsreenu/python_for_image_processing_APEER/blob/master/tutorial122_3D_Unet.ipynb

@author: TaniaKleynhans
"""
import os
import numpy as np
import rasterio
import time
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from patchify import patchify, unpatchify
from sklearn.model_selection import train_test_split
#import tensorflow.keras.api._v2.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate, Conv3DTranspose, BatchNormalization, Dropout, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Activation, Concatenate
from tensorflow.keras.utils import to_categorical
#from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

from tensorflow.python.keras.losses import CategoricalCrossentropy
from tensorflow.python.keras.metrics import CategoricalAccuracy

import tensorflow as tf



def conv_block(input, num_filters):
    x = Conv3D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)   #Not in the original network. 
    x = Activation("relu")(x)

    x = Conv3D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)  #Not in the original network
    x = Activation("relu")(x)

    return x

#Encoder block: Conv block followed by maxpooling
def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPooling3D((2, 2, 1))(x)
    return x, p   


#Decoder block
#skip features gets input from encoder for concatenation
def decoder_block(input, skip_features, num_filters):
    x = Conv3DTranspose(num_filters, (2, 2, 2), strides=(2,2,1), padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


#Build Unet using the blocks
def build_unet(input_shape, n_classes):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024) #Bridge

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    # s1, p1 = encoder_block(inputs, 64)
    # s2, p2 = encoder_block(p1, 128)

    # b1 = conv_block(p2, 256) #Bridge

    # d3 = decoder_block(b1, s2, 128)
    # d4 = decoder_block(d3, s1, 64)

    if n_classes == 1:  #Binary
      activation = 'sigmoid'
    else:
      activation = 'softmax'

    outputs = Conv3D(n_classes, 1, padding="same", activation=activation)(d4)  #Change the activation based on n_classes
    print(activation)

    model = Model(inputs, outputs, name="U-Net")
    return model


def load_training_data(filename):
    """Read bands similar to VZ01."""

    src = rasterio.open(filename)
    img = src.read(1)
    
    data = np.empty([img.shape[0],img.shape[1],6])

    img = src.read(2) # blue
    data[:,:,0] = img * 2.0000E-05 - 0.100000
    img = src.read(3) # green
    data[:,:,1] = img * 2.0000E-05 - 0.100000
    img = src.read(4) # red
    data[:,:,2] = img * 2.0000E-05 - 0.100000
    img = src.read(5) # NIR (similar to band 7 of VZ01 vnir)
    data[:,:,3] = img * 2.0000E-05 - 0.100000
    img = src.read(8) # pan band
    data[:,:,4] = img * 2.0000E-05 - 0.100000
    img = src.read(10)
    mask = img
    img = img * 3.3420E-04 + 0.1
    data[:,:,5] = apptemp(img,  band ='b10')
    mask[mask < 1] = 0
    mask[mask > 0] = 1
    
    for band in range(data.shape[2]):
        data[:,:,band] = data[:,:,band] * mask

    filename_lables = filename[:-8] + 'labels.tif'
    src = rasterio.open(filename_lables)
    img = src.read(1)
    label = img
    
    filename_lables = filename[:-8] + 'qmask.tif'
    src = rasterio.open(filename_lables)
    img = src.read(1)
    qmask = img
    
    filename_lables = filename[:-8] + 'c1bqa.tif'
    src = rasterio.open(filename_lables)
    img = src.read(1)
    c1bqa = img
    
    filename_rgb = filename[:-8] + 'photo.png'
    src = rasterio.open(filename_rgb)
    R = src.read(1)
    G = src.read(2)
    B = src.read(3)
    rgb = np.zeros((R.shape[0], R.shape[1],3))
    rgb[:,:,0] = R/np.nanmax(R)
    rgb[:,:,1] = G/np.nanmax(G)
    rgb[:,:,2] = B/np.nanmax(B)
    rgb = rgb
    
    return data, label, qmask, c1bqa, rgb    


def apptemp(img,  band ='b10'):
    # K1 and K2 constants to be found in MTL.txt file for each band
    if band == 'b10':
        K2 = 1321.0789;
        K1 = 774.8853;
    elif band == 'b11':
        K2 = 1201.1442;
        K1 = 480.8883;
    else:
        print('call function with T = appTemp(radiance, band=\'b10\'')
        return
    temperature = np.divide(K2,(np.log(np.divide(K1,np.array(img)) + 1)))
    temperature = temperature - 273.15  # convert to C
    return temperature


def load_and_format_training_data(filepath):

    cnt = 0
    for file in os.listdir(filepath):
        if '_data' in file:
            try:
                filename = os.path.join(filepath, file)            
                data, label, qmask, c1bqa, rgb = load_training_data(filename) 
                
                if cnt == 0:
                    image = data
                    mask = label
                    cnt+=1
                else:
                    image = np.hstack((image, data)) # (256,256,256) # (1000,79000,6)
                    mask = np.hstack((mask, label))
            except:
                pass
    
             # (256,256,256) # (1000,79000)
    
    n_classes = 7 #np.unique(mask).shape[0]
    
    # create smaller section per image 
    img_patches = patchify(image, (64, 64, 6), step=62)  # (4,4,4,64,64,64) (8, 698, 1, 128, 128, 6)#Step=64 for 64 patches means no overlap
    mask_patches = patchify(mask.astype(int), (64, 64), step=62)  # (4,4,4,64,64,64) (8, 698, 128, 128)

    # img_patches = patchify(image, (128, 128, 6), step=109)  # (4,4,4,64,64,64) (8, 698, 1, 128, 128, 6)#Step=64 for 64 patches means no overlap
    # mask_patches = patchify(mask.astype(int), (128, 128), step=109)  # (4,4,4,64,64,64) (8, 698, 128, 128)

    input_img = np.reshape(img_patches, (-1, img_patches.shape[3], img_patches.shape[4], img_patches.shape[5])) #(64,64,64,64) (5584, 128, 128, 6)
    input_mask = np.reshape(mask_patches, (-1, mask_patches.shape[2], mask_patches.shape[3])) #(64,64,64,64) (5584, 128, 128)
    
    # convert to each channel to rgb for now...
    train_img = np.stack((input_img,)*1, axis=-1) # (64,64,64,64,3) (5584, 128, 128, 6, 1)
    train_img = train_img / np.max(train_img) 
    train_mask = np.stack((input_mask,)*6, axis=-1) #(5584, 128, 128, 6)
    train_mask = np.expand_dims(train_mask, axis=4) # (64,64,64,64,1) (5584, 128, 128, 6, 1)

    
    train_mask_cat = to_categorical(train_mask, num_classes=n_classes)
    X_train, X_test, y_train, y_test = train_test_split(train_img, train_mask_cat, test_size = 0.10, random_state = 0)
    
    return n_classes, X_train, X_test, y_train, y_test 


def plotting_results(history):
    ###
    #plot the training and validation IoU and loss at each epoch
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    acc = history.history['categorical_accuracy']
    val_acc = history.history['val_categorical_accuracy']
    
    plt.subplot(1,2,2)
    plt.plot(epochs, acc, 'y', label='Categorical accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
    plt.savefig('../models/history_plot.png')
    
    
def main_one_gpu(model_filename, data_filepath, batch_size, epochs):

    start_time = time.time()

    n_classes, X_train, X_test, y_train, y_test = load_and_format_training_data(data_filepath)
    
    patches, patch_size_x, patch_size_y, patch_size_z, channels = X_train.shape
    
    input_shape = [patch_size_x, patch_size_y, patch_size_z, channels]
        
        
    model = build_unet(input_shape, n_classes=n_classes)

    LR = 0.0001
    optim = Adam(LR)
    
    model.compile(optimizer=optim, loss=CategoricalCrossentropy(), metrics=CategoricalAccuracy())
    
    print(model.summary())
    print("Patches: ", patches)
    
    #Fit the model
    history=model.fit(X_train, 
              y_train,
              batch_size=batch_size, 
              epochs=epochs,
              verbose=1,
              validation_data=(X_test, y_test))
    
    end_time = time.time()
    print('Total time in min: ',(end_time - start_time)/60)
    
    # save model
    model.save(model_filename)
    
    plotting_results(history)


    return model, history
    
    
model_filename = '../models/sparcs_3D_100epochs_32bs_64patch_1gpu.h5'
data_filepath = '../scenes/'
batch_size = 32
epochs = 100

model, history = main_one_gpu(model_filename, data_filepath, batch_size, epochs)











