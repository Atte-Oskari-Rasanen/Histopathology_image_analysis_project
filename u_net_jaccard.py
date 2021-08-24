#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 16:24:24 2021

@author: atte
"""

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from keras import backend as K
import numpy as np
import os
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import re

print('starting...')
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 3
X_train = np.load('/home/inf-54-2020/experimental_cop/scripts/X_train_size128.npy')
Y_train = np.load('/home/inf-54-2020/experimental_cop/scripts/Y_train_size128.npy')
X_test = np.load('/home/inf-54-2020/experimental_cop/scripts/X_test_size128.npy')
TRAIN_IMG_DIR = "/home/inf-54-2020/experimental_cop/Train_H_Final/Aug_Img/"
M_TRAIN_IMG_DIR = "/home/inf-54-2020/experimental_cop/Train_H_Final/Aug_Mask/"

#TRAIN_IMG_DIR = '/home/atte/kansio/img/'
#M_TRAIN_IMG_DIR ='/home/atte/kansio/img_mask/'

VAL_IMG_DIR = "/home/inf-54-2020/experimental_cop/Val_H_Final/Orginal_unpatched/"
M_VAL_IMG_DIR = "/home/inf-54-2020/experimental_cop/Val_H_Final/Masks/"

n3 = 0
Y_test = []
for root, subdirectories, files in os.walk(M_VAL_IMG_DIR): #tqdm shows the progress bar of the for loop
    #print(root)
    for subdirectory in subdirectories:
    #    print(subdirectory)
        file_path = os.path.join(root, subdirectory)
     #   print(file_path)
        for f in os.listdir(file_path):
            if not f.endswith('.tif'):
                continue
            img_path=file_path + '/' + f   #create first of dic values, i.e the path
            #print(img_path)
            #imagename=ntpath.basename(imagepath)#take the name of the file from the path and save it
            img = imread(img_path)[:,:,:IMG_CHANNELS]
            #sizes_test.append([img.shape[0], img.shape[1]])
            img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
            Y_test.append(img)
            #print(' loop of X_test done!')
Y_test = np.array(Y_test)
np.save('/home/inf-54-2020/experimental_cop/scripts/X_test_size128.npy', Y_test)

def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


def jacard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)  # -1 ultiplied as we want to minimize this value as loss function

################################################################
def simple_unet_model_with_jacard(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
#Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    #s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
    s = inputs

    #Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
     
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
     
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
     
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    #Expansive path 
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
     
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
     
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
     
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
     
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer = 'adam', loss = [jacard_coef_loss], metrics = [jacard_coef])

    model.summary()
    
    return model

jacard_model = simple_unet_model_with_jacard(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

results = jacard_model.fit(X_train, Y_train, batch_size = 16, verbose = 1, epochs = 200, 
                           validation_data = (X_test, Y_test))
cp_save_path = "/home/inf-54-2020/experimental_cop/scripts/Jacard_model.h5"
jacard_model.save(cp_save_path)

