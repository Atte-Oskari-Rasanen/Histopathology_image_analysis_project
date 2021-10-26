#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 11:11:48 2021

@author: atte
"""

import os
import random
import numpy as np
from tqdm import tqdm 
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
seed = 42
np.random.seed = seed
import PIL
from PIL import Image, ImageOps
import cv2
from keras.utils import normalize

import tensorflow as tf
import os
import random
import numpy as np
from tensorflow import keras
from tifffile import imsave
import ntpath
from skimage import io


IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 3

TRAIN_IMG_DIR = "/home/inf-54-2020/experimental_cop/Train_H_Final/Full_Aug_Img/"
M_TRAIN_IMG_DIR = "/home/inf-54-2020/experimental_cop/Train_H_Final/Full_Aug_Mask/"

#`TRAIN_IMG_DIR = '/home/atte/kansio/img/'
#M_TRAIN_IMG_DIR ='/home/atte/kansio/img_mask/'

VAL_IMG_DIR = "/home/inf-54-2020/experimental_cop/Val_H_Final/Images/"
M_VAL_IMG_DIR = "/home/inf-54-2020/experimental_cop/Val_H_Final/Masks/"
KAGGLE_DIR = "/home/inf-54-2020/experimental_cop/kaggle_data/"


train_ids = next(os.walk(KAGGLE_DIR))[1] #returns all sub dirs found within this dir 
m_train_ids = next(os.walk(KAGGLE_DIR))[1] #returns all sub dirs found within this dir 


# cp_save_path = "/home/inf-54-2020/experimental_cop/scripts/New_model_bs128.h5"
# model_segm = keras.models.load_model(cp_save_path)

im_path = "/home/inf-54-2020/experimental_cop/Train_H_Final/Train/"

TRAIN_PATH = '/home/inf-54-2020/experimental_cop/kaggle_data/'
# X_train1 = []
# Y_train1 = []

# for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):   
#     path = TRAIN_PATH + id_
#     img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]  
#     img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
#     #print(img.shape)
#     X_train1.append(img)  #Fill empty X_train with values from img
#     mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
#     for mask_file in next(os.walk(path + '/masks/'))[2]:
#         mask_ = imread(path + '/masks/' + mask_file)
#         mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',  
#                                       preserve_range=True), axis=-1)
#         mask = np.maximum(mask, mask_)  
            
#     Y_train1.append(mask)
# np.save('/home/inf-54-2020/experimental_cop/scripts/kagl_X_train_size512_pcis.npy', X_train1)
# np.save('/home/inf-54-2020/experimental_cop/scripts/kagl_Y_train_size512_pcis.npy', Y_train1)

X_train1 = np.load('/home/inf-54-2020/experimental_cop/scripts/kagl_X_train_size512_pcis.npy')
Y_train1 = np.load('/home/inf-54-2020/experimental_cop/scripts/kagl_Y_train_size512_pcis.npy')

def start_points(size, split_size, overlap=0):
    points = [0]
    stride = int(split_size * (1-overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            points.append(size - split_size)
            break
        else:
            points.append(pt)
        counter += 1
    return points

n = 0
def patches(filedir,XY_list):
    for imagefile in os.listdir(filedir):  #to go through files in the specific directory
        #print(os.listdir(directory))
        imagepath=filedir + imagefile
        if imagefile.endswith('.png'):
            img = Image.open(imagepath)
            #print(imagefile)
            img = np.asarray(img)
            #print(img.shape)
            try:
                img_h, img_w, _ = img.shape
            except ValueError:
                pass
            global img_w, img_h #to avoid UnboundLocalError at X_points
            #img = np.resize(img, (500,500))
            split_width = 512 #244
            split_height = 512 #244
        
            X_points = start_points(img_w, split_width, 0.1)
            Y_points = start_points(img_h, split_height, 0.1)
            #print(Y_points.shape)
            
            #Split the image
            for i in Y_points:
                for j in X_points:
                    #sometimes some images may be corrupted and thus their array size and shape is 0,
                    #in these scenarios an error occurs. This images may be skipped via try except
                    try:
                        split = img[i:i+split_height, j:j+split_width]
                        #split = np.expand_dims(split, 0)
                    except IndexError:
                        pass
                    #print('==================')
                    XY_list.append(split)
    print(XY_list[0].shape)

    XY_list = np.asarray(XY_list).astype(object)
    print(XY_list.shape)
    return XY_list


def patches_mask(filedir,XY_list):
    for imagefile in os.listdir(filedir):  #to go through files in the specific directory
        #print(os.listdir(directory))
        imagepath=filedir + imagefile
        if imagefile.endswith('.png'):
            img = io.imread(imagepath, as_gray=True)
            #print(imagefile)
            img = np.asarray(img)
            #print(img.shape)
            try:
                img_h, img_w, _ = img.shape
            except ValueError:
                pass
            global img_w, img_h #to avoid UnboundLocalError at X_points
            #img = np.resize(img, (500,500))
            split_width = 512 #244
            split_height = 512 #244
        
            X_points = start_points(img_w, split_width, 0.1)
            Y_points = start_points(img_h, split_height, 0.1)
            #print(Y_points.shape)
            
            
            #Split the image
            for i in Y_points:
                for j in X_points:
                    #sometimes some images may be corrupted and thus their array size and shape is 0,
                    #in these scenarios an error occurs. This images may be skipped via try except
                    try:
                        split = img[i:i+split_height, j:j+split_width]
                        split = np.expand_dims(split, 2)
                        #split = np.expand_dims(split, 0)
                        #split= split.astype(object)
                        #this may need to be done as well so that you get the channel dim and hte im number dim:
                    except IndexError:
                        pass
                    #if split.shape==(1,512,512):
                    XY_list.append(split)
    print(XY_list[0].shape)
    #print(XY_list.shape)
    # XY_list = np.reshape(XY_list, (512, 512, 1))
    XY_list = np.asarray(XY_list).astype(object)
    # print(XY_list.shape)
    # imgs_no = len(XY_list)
    # XY_list = XY_list.reshape(imgs_no, 512,512,1) #aiemmas reshape oli vaan 512,512 #-1 means an unknwon dim, numpy calculates this
    # #XY_list = np.expand_dims(XY_list, 3)
    # #XY_list = np.expand_dims(XY_list, 0)
    print(XY_list.shape)
    return XY_list

X_train2 = []
Y_train2 = []
X_train2 = patches(TRAIN_IMG_DIR, X_train2)
Y_train2 = patches_mask(M_TRAIN_IMG_DIR, Y_train2)
#Y_train2 = np.asarray(Y_train2)
print(X_train1.shape)
print(Y_train1.shape)
print('============')
print(X_train2.shape)
print(Y_train2.shape)

X_train = np.concatenate((X_train1,X_train2), axis = 0)
Y_train = np.concatenate((Y_train1,Y_train2), axis = 0)
#save as pcis - patches created in script
np.save('/home/inf-54-2020/experimental_cop/scripts/kagl_own_X_train_size512_pcis.npy', X_train)
np.save('/home/inf-54-2020/experimental_cop/scripts/kagl_own_Y_train_size512_pcis.npy', Y_train)

#print(X_train.shape)


#Build the model
inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

#Contraction path
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
c1 = tf.keras.layers.Dropout(0.1)(c1)
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
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

################################
#Modelcheckpoint
cp_save_path = "/home/inf-54-2020/experimental_cop/scripts/Model_512_pcis.h5"


callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='logs')]


model.save(cp_save_path)
checkpointer = tf.keras.callbacks.ModelCheckpoint(cp_save_path, verbose=1, save_best_only=True)
#model.save_weights(cp_save_path)
print('Model built and saved, now fitting it...')
#results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=200, callbacks=callbacks)

