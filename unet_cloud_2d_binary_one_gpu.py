# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 15:27:41 2022

@author: TaniaKleynhans
"""
import os
import numpy as np
import rasterio
import time
import matplotlib.pyplot as plt
from patchify import patchify, unpatchify
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.losses import BinaryCrossentropy
from tensorflow.python.keras.metrics import BinaryAccuracy
from tensorflow.keras.optimizers import Adam
from osgeo import gdal, gdalconst


def load_training_data(filename, bands):
    """Read bands similar to VZ01.

    Args:
        filename (str): 
    """
    

    src = rasterio.open(filename)
    img = src.read(1)
    
    data = np.empty([img.shape[0],img.shape[1],bands])

    img = src.read(2) # blue
    data[:,:,0] = img * 2.0000E-05 - 0.100000
    img = src.read(3) # green
    data[:,:,1] = img * 2.0000E-05 - 0.100000
    img = src.read(4) # red
    data[:,:,2] = img * 2.0000E-05 - 0.100000
    img = src.read(5) # NIR (similar to band 7 of VZ01 vnir)
    data[:,:,3] = img * 2.0000E-05 - 0.100000
    img = src.read(10)
    mask = img
    img = img * 3.3420E-04 + 0.1
    
    mask[mask < 1] = 0
    mask[mask > 0] = 1
    
    if bands == 6:
        img = src.read(8) # pan band
        data[:,:,4] = img * 2.0000E-05 - 0.100000
        data[:,:,5] = apptemp(img,  band ='b10')
    else:
        data[:,:,4] = apptemp(img,  band ='b10')
    
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
    rgb = rgb *1.5
    
    # normalize data with values accross all scenes (subset max values skews data)
    if bands == 6:
        max_training = [1.2107, 1.2107, 1.2107, 1.2107, 1.03068, 91.33669913673828]
    else:
        max_training = [1.2107, 1.2107, 1.2107, 1.2107, 91.33669913673828]
    
    for i in range(data.shape[2]):
        data[:,:,i] = data[:,:,i] / max_training[i] 
    
    return data, label, qmask, c1bqa, rgb    


def apptemp(img,  band ='b10'):
    """Convert radiance to apparent temperature."""
    # K1 and K2 constants to be found in MTL.txt file for each band
    if band == 'b10':
        K2 = 1321.0789
        K1 = 774.8853
    elif band == 'b11':
        K2 = 1201.1442
        K1 = 480.8883
    else:
        print('call function with T = appTemp(radiance, band=\'b10\'')
        return
    temperature = np.divide(K2,(np.log(np.divide(K1,np.array(img)) + 1)))
    temperature = temperature - 273.15  # convert to C
    return temperature


def load_and_format_training_data(filepath, classification, xy=64, steps=64, bands=6):
    """Load training data and convert to correct format."""
    if classification == 'snow':
        class_num = [3]
    elif classification == 'cloud':
        class_num = [5]
    elif classification == 'shadow':
        class_num = [0, 1]

    cnt = 0
    # plt.figure()
    for file in os.listdir(filepath):
        if '_data' in file:
            try:
                filename = os.path.join(filepath, file)            
                data, label, qmask, c1bqa, rgb = load_training_data(filename, bands) 
                # mask_label = np.copy(label)
                for idx in class_num:
                    label[label == idx] = 10
                label[label != 10] = 0
                label[label == 10] = 1
                   
                if cnt == 0:
                    image = data
                    mask = label
                    cnt+=1
                else:
                    image = np.hstack((image, data)) # (256,256,256) # (1000,79000,6)
                    mask = np.hstack((mask, label))
                    cnt+=1

            except:
                pass
    
    print(f"Number of scenes: {cnt}")
    
    n_classes = 2 #np.unique(mask).shape[0]
    
    # create smaller section per image 
    img_patches = patchify(image, (xy, xy, bands), step=steps)  # Step=64 for 64 patches means no overlap
    mask_patches = patchify(mask.astype(int), (xy, xy), step=steps)

    input_img = np.reshape(img_patches, (-1, img_patches.shape[3], img_patches.shape[4], img_patches.shape[5]))
    input_mask = np.reshape(mask_patches, (-1, mask_patches.shape[2], mask_patches.shape[3]))

    X_train, X_test, y_train, y_test = train_test_split(input_img, input_mask, test_size = 0.10, random_state = 2)

    return n_classes, X_train, X_test, y_train, y_test 


def conv_block(input, num_filters):
    """Define convolutional block."""
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


def encoder_block(input, num_filters):
    """Define encoder block."""
    x = conv_block(input, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p


def decoder_block(input, skip_features, num_filters):
    """Define decoder block."""
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


def build_unet(input_shape, block=2):
    """Build the unet model."""
    inputs = Input(input_shape) # height, width, channels ... e.g. 128 x 128 x 6

    if block==2:
        s1, p1 = encoder_block(inputs, 64)
        s2, p2 = encoder_block(p1, 128)
        
        b1 = conv_block(p2, 256) #Bridge
        
        d1 = decoder_block(b1, s2, 128)
        d2 = decoder_block(d1, s1, 64)
        
        outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d2)  # sigmoid for binary classification
         
    elif block==3:
        s1, p1 = encoder_block(inputs, 64)
        s2, p2 = encoder_block(p1, 128)
        s3, p3 = encoder_block(p2, 256)
    
        b1 = conv_block(p3, 512)
    
        d1 = decoder_block(b1, s3, 256)
        d2 = decoder_block(d1, s2, 128)
        d3 = decoder_block(d2, s1, 64)
        
        outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d3)  # sigmoid for binary classification
        
    elif block==4:
        
        s1, p1 = encoder_block(inputs, 64)
        s2, p2 = encoder_block(p1, 128)
        s3, p3 = encoder_block(p2, 256)
        s4, p4 = encoder_block(p3, 512)
    
        b1 = conv_block(p4, 1024)
    
        d1 = decoder_block(b1, s4, 512)
        d2 = decoder_block(d1, s3, 256)
        d3 = decoder_block(d2, s2, 128)
        d4 = decoder_block(d3, s1, 64)
    
        outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)  # sigmoid for binary classification

    model = Model(inputs, outputs, name="U-Net")
    return model


def plotting_results(history):
    """Plot the training and validation IoU and loss at each epoch."""
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
    
    acc = history.history['binary_accuracy']
    val_acc = history.history['val_binary_accuracy']
    
    plt.subplot(1,2,2)
    plt.plot(epochs, acc, 'y', label='Binary accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def plot_full_scene(model, test_filename, xy=64, bands = 6):
    """Predict feature from full scene input."""
    #Break the large image (volume) into patches of same size as the training images (patches)
    full_image, label, qmask, c1bqa, rgb = load_training_data(test_filename, bands = 6)
    
    boundary = 5
    steps = xy - (boundary * 2)  # to be able to remove boundary created by padding
    patches = patchify(full_image, (xy, xy, 6), step=steps)  #Step=256 for 256 patches means no overlap
    # patches_new = np.reshape(patches, (-1, patches.shape[3], patches.shape[4], patches.shape[5]))

    print(full_image.shape)
    print(patches.shape)
    # print(patches_new.shape)
    
    reconstructed_image = np.zeros((full_image.shape[0], full_image.shape[1]))
    
    i_start_row = boundary

    for i in range(patches.shape[0]):
        j_start_col = boundary
        for j in range(patches.shape[1]):
            single_patch = patches[i,j,0,:,:,:]
            single_patch_input = np.expand_dims(single_patch, axis=0)
            single_patch_prediction = model.predict(single_patch_input)
            
            # remove boundaries
            single_patch_prediction_no_boundaries = single_patch_prediction[0, boundary:-boundary, boundary:-boundary, 0]
            
            # add prediction to image
            reconstructed_image[i_start_row : i_start_row + steps, j_start_col : j_start_col + steps] = single_patch_prediction_no_boundaries
            
            j_start_col = j_start_col + steps
        i_start_row = i_start_row + steps     
    
    fig = plt.figure()
    plt.subplot(2,3,1)
    plt.imshow(rgb)
    plt.axis('off')
    plt.title('Scaled color')
    plt.show()
    plt.subplot(2,3,2)
    plt.imshow(label, cmap='Dark2', vmin=0, vmax=7)
    plt.axis('off')
    plt.title('Truth classification')
    plt.show()
    plt.subplot(2,3,3)
    ax2 = plt.imshow(reconstructed_image) #, cmap='Dark2', vmin=0, vmax=7)
    plt.axis('off')
    plt.title('Predicted classification')
    plt.colorbar()
    #cbar = fig.colorbar(ax2, ticks=[0,1,2,3,4,5,6,7])
    #cbar.ax.set_yticklabels(['Cloud Shadow', 'Cloud Shadow over Water', 'Water',
    #                          'Ice/Snow','Land','Clouds','Flooded','None'])
    plt.show()
    scaled = np.where(reconstructed_image > 0.85, 1, 0)
    plt.subplot(2,3,4)
    plt.imshow(scaled)
    plt.axis('off')
    plt.title('Threshold = 0.85')
    plt.show()
    scaled = np.where(reconstructed_image > 0.9, 1, 0)
    plt.subplot(2,3,5)
    plt.imshow(scaled)
    plt.axis('off')
    plt.title('Threshold = 0.9')
    plt.show()
    scaled = np.where(reconstructed_image > 0.92, 1, 0)
    plt.subplot(2,3,6)
    plt.imshow(scaled)
    plt.axis('off')
    plt.title('Threshold = 0.92')
    plt.show()


def stackImg1Img2(img1_filename, img2_filename, dst_filename):
    
    #img1 = geotif image which resolution you want to keep
    #img2 = geotif image to be resampled and matched to img1 - this will be the output image
    
    # Source
    src_filename = img2_filename
    src = gdal.Open(src_filename, gdalconst.GA_ReadOnly)
    src_proj = src.GetProjection()
    #src_geotrans = src.GetGeoTransform()
    
    # We want a section of source that matches this:
    match_filename = img1_filename
    match_ds = gdal.Open(match_filename, gdalconst.GA_ReadOnly)
    match_proj = match_ds.GetProjection()
    match_geotrans = match_ds.GetGeoTransform()
    wide = match_ds.RasterXSize
    high = match_ds.RasterYSize
    
    try:
        dst = gdal.GetDriverByName('GTiff').Create(dst_filename, wide, high, 5, gdalconst.GDT_Float32)
        dst.SetGeoTransform( match_geotrans )
        dst.SetProjection(match_proj)
        # Do the work
        gdal.ReprojectImage(src, dst, src_proj, match_proj, gdalconst.GRA_Bilinear)
        
        del dst # Flush
        
    except:
        pass


def plot_predict_new_scene(model, test_filename, xy=64, bands=6):
    """Predict feature from full scene input."""
    
    src = rasterio.open(test_filename)
    base_filename = test_filename
    img = src.read(1)
    
    data = np.empty([img.shape[0],img.shape[1],6])
    for file in os.listdir(os.path.dirname(test_filename)):
        print(file)
        if "_B2" in file:
            test_filename = os.path.join(os.path.dirname(test_filename), file)
            src = rasterio.open(test_filename)
            img = src.read(1) # blue
            data[:,:,0] = img.astype(float) * 2.0000E-05 - 0.100000  
        if "_B3" in file:
            test_filename = os.path.join(os.path.dirname(test_filename), file)
            src = rasterio.open(test_filename)
            img = src.read(1) # green
            data[:,:,1] = img.astype(float) * 2.0000E-05 - 0.100000
        if "_B4" in file:
            test_filename = os.path.join(os.path.dirname(test_filename), file)
            src = rasterio.open(test_filename)
            img = src.read(1) # red
            data[:,:,2] = img.astype(float) * 2.0000E-05 - 0.100000
        if "_B5" in file:
            test_filename = os.path.join(os.path.dirname(test_filename), file)
            src = rasterio.open(test_filename)
            img = src.read(1) # NIR (similar to band 7 of VZ01 vnir)
            data[:,:,3] = img.astype(float) * 2.0000E-05 - 0.100000
        if bands==6:
            if "_B8.TIF" in file:
                # resample data to 30 m landsat file
                test_filename = os.path.join(os.path.dirname(test_filename), file)
                dst_filename = test_filename[0:-4] + "_resampled.TIF"
                stackImg1Img2(base_filename, test_filename, dst_filename)
                src = rasterio.open(dst_filename)
                img = src.read(1) # pan band
                data[:,:,4] = img.astype(float) * 2.0000E-05 - 0.100000
            if "_B10" in file:
                test_filename = os.path.join(os.path.dirname(test_filename), file)
                src = rasterio.open(test_filename)
                img = src.read(1) # thermal
                img = img.astype(float) * 3.3420E-04 + 0.1
                data[:,:,5] = apptemp(img,  band ='b10')
        else:
            if "_B10" in file:
                test_filename = os.path.join(os.path.dirname(test_filename), file)
                src = rasterio.open(test_filename)
                img = src.read(1) # thermal
                img = img.astype(float) * 3.3420E-04 + 0.1
                data[:,:,4] = apptemp(img,  band ='b10')
            
    rgb = data[:,:,0:3] *1.5
    
    boundary = 5
    steps = xy - (boundary * 2)  # to be able to remove boundary created by padding
    patches = patchify(data, (xy, xy, bands), step=steps)  #Step=256 for 256 patches means no overlap
    # patches_new = np.reshape(patches, (-1, patches.shape[3], patches.shape[4], patches.shape[5]))

    print(data.shape)
    print(patches.shape)
    # print(patches_new.shape)
    
    reconstructed_image = np.zeros((data.shape[0], data.shape[1]))
    
    i_start_row = boundary

    for i in range(patches.shape[0]):
        j_start_col = boundary
        for j in range(patches.shape[1]):
            single_patch = patches[i,j,0,:,:,:]
            single_patch_input = np.expand_dims(single_patch, axis=0)
            single_patch_prediction = model.predict(single_patch_input)
            
            # remove boundaries
            single_patch_prediction_no_boundaries = single_patch_prediction[0, boundary:-boundary, boundary:-boundary, 0]
            
            # add prediction to image
            reconstructed_image[i_start_row : i_start_row + steps, j_start_col : j_start_col + steps] = single_patch_prediction_no_boundaries
            
            j_start_col = j_start_col + steps
        i_start_row = i_start_row + steps     
    
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(rgb)
    plt.axis('off')
    plt.title('Scaled color')
    plt.show()
    plt.subplot(1,3,2)
    plt.imshow(reconstructed_image) #, cmap='Dark2', vmin=0, vmax=7)
    plt.axis('off')
    plt.title('Predicted classification')
    plt.colorbar()
    plt.show()
    scaled = np.where(reconstructed_image > 0.85, 1, 0)
    plt.subplot(1,3,3)
    plt.imshow(scaled)
    plt.axis('off')
    plt.title('Threshold = 0.85')
    plt.show()


def display_per_class_accuracy(model, X_test, y_test):
    
    y_pred = model.predict(X_test)
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred != 1] = 0
    
    print(classification_report(y_test.flatten(), y_pred.flatten()))

def main(model_filename, data_filepath, batch_size, epochs, classification, xy, steps, bands, block):
    """Run application."""

    print("batch_size: ", batch_size)
    start_time = time.time()

    n_classes, X_train, X_test, y_train, y_test = load_and_format_training_data(data_filepath, classification, xy, steps, bands)
    
    patches, patch_size_x, patch_size_y, channels = X_train.shape

    input_shape = [patch_size_x, patch_size_y, channels]
    print("Input shape: ", input_shape)

    model = build_unet(input_shape, block)

    # define learning rate
    LR = 0.0001
    optim = Adam(LR)
    # Complie model
    model.compile(optimizer=optim, loss=BinaryCrossentropy(), metrics=BinaryAccuracy())
    print(model.summary())
    print("Patches: ", patches)
    
    # Fit the model
    history=model.fit(X_train, 
              y_train,
              batch_size=batch_size, 
              epochs=epochs,
              verbose=1,
              validation_data=(X_test, y_test))
    
    display_per_class_accuracy(model, X_test, y_test)
    
    end_time = time.time()
    print('Total time in min: ',(end_time - start_time)/60)

    # save model
    model.save(model_filename)

    plotting_results(history)

    return model, history


if __name__ == "__main__":
    batch_size = 32
    epochs = 50
    bands = 5  # number of bands to use for training (6 includes Pan band)
    block = 3  # number of encoder-decoder blocks
    xy = 64  # patch size (larger needs more memory)
    steps = 50 # if steps are same as xy, then patches will have no overlap. This is not neccesary, but overlap creates more data to train on.
    classification = 'snow'  # 'cloud', 'shadow'
    test_full_scene = 0
    test_outside_scene = 0
    run_model = 1
    
    if run_model == 1:
        data_filepath = '/home/tkleynhans/hydrosat/data/scenes/'
        model_filepath = '/home/tkleynhans/hydrosat/data/models'
        model_fname = f'sparcs_2D_{epochs}epochs_{batch_size}bs_{classification}_{xy}patch_{steps}step_{bands}bands_{blocks}blocks_1gpu.h5'
        model_filename = os.path.join(model_filepath, model_fname)
        model, history = main(model_filename, data_filepath, batch_size, epochs, classification, xy, steps, bands, block)
    
    if test_full_scene == 1:
        # testing on full scene
        scene = "LC81480352013195LGN00_32_data.tif" # snow scene
        model_filename = '/home/tkleynhans/hydrosat/data/models/sparcs_2D_100epochs_32bs_64patch_snow_64patch_50step_1gpu.h5'
        model = load_model(model_filename, compile=False)
        filepath = '/home/tkleynhans/hydrosat/data/scenes/'
        test_filename = os.path.join(filepath, scene)
        plot_full_scene(model, test_filename, xy)
    
    if test_outside_scene == 1:
        # test on scene not part of sparcs
        start_time = time.time()
        model_filename = '/home/tkleynhans/hydrosat/data/models/sparcs_2D_100epochs_32bs_64patch_snow_64patch_50step_1gpu.h5'
        model = load_model(model_filename, compile=False)
        scene = "LC08_L1TP_016030_20160303_20200907_02_T1_B1.TIF"
        filepath = '/home/tkleynhans/hydrosat/data/landsat/LC08_L1TP_016030_20160303_20200907_02_T1'
        test_filename = os.path.join(filepath, scene)
        plot_predict_new_scene(model, test_filename, xy=64, bands=5)
        end_time = time.time()
        print('Total time in min: ',(end_time - start_time)/60)
    
    
    
    
    
