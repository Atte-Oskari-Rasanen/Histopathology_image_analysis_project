#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 11:14:35 2021

@author: atte
"""
import tensorflow as tf
import os
import random
import numpy as np
import cv2
from tqdm import tqdm 

from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from PIL import Image

# =============================================================================
#   File "U_net_v2.py", line 93, in <module>
#     Y_train[i] = mask
# ValueError: could not broadcast input array from shape (1792,1792,1792) into shape (1792,1792,1)
# (tf-2) [inf-54-2020@localhost experimental_cop]$ 
# =============================================================================

seed = 42
np.random.seed = seed

IMG_WIDTH = 224  #find the ideal ones so that all the images can be resized into these dimensions!
IMG_HEIGHT = 224
IMG_CHANNELS = 3
plots_path = "/home/inf-54-2020/experimental_cop/"

TRAIN_IMG_DIR = "/home/inf-54-2020/experimental_cop/H_final/Images/"
M_TRAIN_IMG_DIR = "/home/inf-54-2020/experimental_cop/H_final/Masks/"

VAL_IMG_DIR = "/home/inf-54-2020/experimental_cop/Val_H_Final/Images/"

#use the names of the images as the ids
train_ids = next(os.walk(TRAIN_IMG_DIR))[1]
print("Train_ids content:")
test_ids = next(os.walk(VAL_IMG_DIR))[1]

list = os.listdir(TRAIN_IMG_DIR)
print(len(list))
X_train = np.zeros((len(list), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

#Y is the var we are trying to predicts based on X
Y_train = np.zeros((len(list), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
print('Resizing training images and masks')

i=1
for i, file in enumerate(os.listdir(TRAIN_IMG_DIR)):
    
#for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):   

    path=TRAIN_IMG_DIR
    path2 = M_TRAIN_IMG_DIR
    img = imread(path + file)[:,:,:IMG_CHANNELS]  
    #print(path + file)
    #img = image.load_img(path + file , target_size=(IMG_HEIGHT, IMG_WIDTH))
    #print("img:")
    #print(img)
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    #print(img)
    #print(img.shape)
    
    X_train[i] = img  #Fill empty X_train with values from img
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    #print("made up mask")
    #print(mask.shape)
    for mask_file in next(os.walk(path2))[2]:
        mask_ = cv2.imread(path + mask_file)[:,:,:1]  
        #mask_ = Image.fromarray(mask_,)
        #mask_ = mask_.convert("RGB")
        #mask_ = np.asarray(mask_)

        #print(mask_.shape)
        mask_ = resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        #print(mask_.size)
        #mask_ = mask_.reshape([IMG_HEIGHT, IMG_WIDTH, 1])
        #mask_ = np.array(mask_, dtype=np.bool)
        #print(mask_)
# =============================================================================
#         mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',  
#                                       preserve_range=True), axis=0)
#         print(mask_.shape)
# =============================================================================
        #print("Imported mask")
        #print(mask_)
        #print(mask_.shape)
        mask = np.maximum(mask, mask_)  #At every pixel look for the max pixel value and create the mask based on that
    #print(mask.size)
    #print(type(mask))

    #print(type(Y_train))
    Y_train[i] = mask

# test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Resizing test images') 

for i, file in enumerate(os.listdir(VAL_IMG_DIR)):
#for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = VAL_IMG_DIR
    img = imread(path + file)[:,:,:IMG_CHANNELS]
    #print(path + file)
    #img = image.load_img(path + file , target_size=(IMG_HEIGHT, IMG_WIDTH))
    #print("img:")
    #print(img)
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    #img = image.load_img(path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[i] = img
    
    #print(img.shape)
    print('sizes_test:')
    print(sizes_test)


print('Done!')

#image_x = random.randint(0, len(train_ids))
#imshow(X_train[image_x])
#plt.savefig(plots_path + 'X_train', bbox_inches='tight')
#plt.show()
#plt.savefig(plots_path + 'Y_train', bbox_inches='tight')
#imshow(np.squeeze(Y_train[image_x]))
#plt.show()




#Build the model
inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)) #Inputs layers. Contains lots of layers used in Keras
#so the input is the image with its properties i.e. height, width, and the number of channels


#the layers only take in floating point values between 0-1. So the input pixel values of 8bit needs to be 
#converted into floats. Divide by 255 since all pixel values in the pic range between 0-255.

s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

#Next we work on the convolutional layers (based on the example image)
#In a neural network, the activation function is responsible for transforming 
#the summed weighted input from the node into the activation of the node or 
#output for that input. -- we use relu will output the input directly if it 
#is positive, otherwise, it will output zero. A default activation function

#kernel_initializer - need to start with some weights that the network then adjusts. 
#kernel initalizer defines the intial weight values. he_normal is a truncated normal distr (centered around 0)
#Other ones can be tried out as well.

#padding=same means that we want out input and output images be the same. inputs at the end of the code
#mean that we are applying this layer to the inputs variable.

#Contraction path
#conv2=downsampling
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
c1 = tf.keras.layers.Dropout(0.1)(c1) #drop out 10% of the c1 layers
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
 
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
 
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
 
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

#Expansive path 
#conv2dtranspose=upsampling
u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
 
u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
 
u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
 
u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
 
outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
 
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
#optimisers contain backpropagation algorithms. Since this is a binary classificatio nwe use binary_crossentropy
#as loss function. Optimiser tries to minimise the loss funciton, once it finds the minimum, then it stops
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

################################
#Modelcheckpoint - to save the model
checkpointer = tf.keras.callbacks.ModelCheckpoint('model_H_nuclei.h5', verbose=1, save_best_only=True)

callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='experimental')]

results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=25, callbacks=callbacks)

####################################

idx = random.randint(0, len(X_train))


preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

 
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)


# Perform a sanity check on some random training samples
ix = random.randint(0, len(preds_train_t))
imshow(X_train[ix])
#plt.show()
plt.imsave('X_train.jpg', prediction_image, cmap='gray')

imshow(np.squeeze(Y_train[ix]))
#plt.show()
plt.imsave('Y_train.jpg', prediction_image, cmap='gray')

imshow(np.squeeze(preds_train_t[ix]))
plt.imsave('preds_train.jpg', prediction_image, cmap='gray')

# Perform a sanity check on some random validation samples
ix = random.randint(0, len(preds_val_t))
imshow(X_train[int(X_train.shape[0]*0.9):][ix])
plt.imsave('X_train2.jpg', prediction_image, cmap='gray')

#plt.show()
imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]))
plt.imsave('Y_train2.jpg', prediction_image, cmap='gray')

#plt.show()
imshow(np.squeeze(preds_val_t[ix]))
plt.imsave('preds_train2.jpg', prediction_image, cmap='gray')

#plt.show()