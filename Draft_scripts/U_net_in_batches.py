#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 14:18:45 2021

@author: atte
"""
#model script:https://www.machinecurve.com/index.php/2020/04/06/using-simple-generators-to-flow-data-from-file-with-keras/
import numpy as np
import pandas as pd

from random import randint

import matplotlib.pyplot as plt
# plt.style.use('seaborn-white')
# import seaborn as sns
# sns.set_style("white")

from sklearn.model_selection import train_test_split
from skimage.transform import resize
import tensorflow as tf

from keras import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout
from tensorflow.keras.models import Model, load_model,Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras import regularizers

from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import schedules
from keras import Model

from tqdm import tqdm
import os
from skimage.io import imread, imshow
from skimage.transform import resize
# matplotlib.use('Agg')
import cv2
import matplotlib.pyplot as plt
import sys 
from keras import train_generator, validation_generator, fit_generator, train_datagen, test_datagen
seed = 42
np.random.seed = seed

# #you need two functions for X_train and Y_train. One for importing kaggle, one for importing own
# #img dims need to be divisible by 32
IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 3
# cp_save_path = '/home/inf-54-2020/experimental_cop/scripts/working_models/kaggle_model'
# model_segm = keras.models.load_model(cp_save_path)

# own_img_dirs = ['/home/inf-54-2020/experimental_cop/Train_H_Final/Aug_Img/OneThird/', '/home/inf-54-2020/experimental_cop/Train_H_Final/Aug_Img/TwoThird/', '/home/inf-54-2020/experimental_cop/Train_H_Final/Aug_Img/ThirdThird/']
# own_masks_dirs = ['/home/inf-54-2020/experimental_cop/Train_H_Final/Aug_Mask/OneThird/', '/home/inf-54-2020/experimental_cop/Train_H_Final/Aug_Mask/TwoThird/', '/home/inf-54-2020/experimental_cop/Train_H_Final/Aug_Mask/ThirdThird/']
# own_imgs_part_a = own_img_dirs[0]
# own_imgs_part_b = own_img_dirs[1]
# own_imgs_part_b = own_img_dirs[2]

# own_masks_part_a = own_masks_dirs[0]
# own_masks_part_b = own_masks_dirs[1]
# own_masks_part_b = own_masks_dirs[2]


# X_train_k = np.load('/home/inf-54-2020/experimental_cop/scripts/np_datas/X_Dataset_k_s512.npy')
# Y_train_k = np.load('/home/inf-54-2020/experimental_cop/scripts/np_datas/Y_Dataset_k_s512.npy')

# X_train_a = np.load('/home/inf-54-2020/experimental_cop/scripts/np_datas/X_Dataset_a_s512.npy')
# Y_train_a = np.load('/home/inf-54-2020/experimental_cop/scripts/np_datas/Y_Dataset_a_s512.npy')
# X_train_b = np.save('/home/inf-54-2020/experimental_cop/scripts/np_datas/X_Dataset_b_s512.npy')
# Y_train_b = np.save('/home/inf-54-2020/experimental_cop/scripts/np_datas/Y_Dataset_b_s512.npy')
# X_train_c = np.load('/home/inf-54-2020/experimental_cop/scripts/np_datas/X_Dataset_c_s512.npy')
# Y_train_c = np.load('/home/inf-54-2020/experimental_cop/scripts/np_datas/Y_Dataset_c_s512.npy')

# X_train_list = [X_train_k, X_train_a, X_train_b, X_train_c]
# Y_train_list = [Y_train_k, Y_train_a, Y_train_a, Y_train_c]

# X_train_generator = train_datagen.flow_from_directory(
#         'data/train',
#         target_size=(512, 512),
#         batch_size=32,
#         class_mode='binary')

# Y_train_generator = train_datagen.flow_from_directory(
#         'data/train',
#         target_size=(512, 512),
#         batch_size=32,
#         class_mode='binary')

# # same es the train_generator    
# val_generator = test_datagen.flow_from_directory(
#         'data/validation',
#         target_size=(150, 150),
#         batch_size=32,
#         class_mode='binary')
#Build the model
inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

#Contraction path
c1 = tf.keras.layers.Conv2D(16*2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16*2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

c2 = tf.keras.layers.Conv2D(32*2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32*2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
 
c3 = tf.keras.layers.Conv2D(64*2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64*2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
 
c4 = tf.keras.layers.Conv2D(128*2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128*2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
 
c5 = tf.keras.layers.Conv2D(256*2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256*2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

#Expansive path 
u6 = tf.keras.layers.Conv2DTranspose(128*2, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(128*2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128*2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
 
u7 = tf.keras.layers.Conv2DTranspose(64*2, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(64*2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64*2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
 
u8 = tf.keras.layers.Conv2DTranspose(32*2, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(32*2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32*2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
 
u9 = tf.keras.layers.Conv2DTranspose(16*2, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
c9 = tf.keras.layers.Conv2D(16*2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16*2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
 
outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
 
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
#model = tf.keras.Model(inputs=[input_layer], outputs=[output_layer]) #putting things into [] may have caused an issue since they should be tuple

# opt = SGD(lr=0.01, momentum=0.9, clipnorm=1.0)
# opt = keras.optimizers.Adam(learning_rate=0.01)

#model = Model(input_layer, output_layer)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

# early_stopping = EarlyStopping(patience=10, verbose=1)
# model_checkpoint = ModelCheckpoint("./keras.model", save_best_only=True, verbose=1)
# reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1)

# epochs = 200
# batch_size = 32

# history = model.fit(X_train, Y_train,
#                     validation_data=[x_valid, y_valid], 
#                     epochs=epochs,
#                     batch_size=batch_size,
#                     callbacks=[early_stopping, model_checkpoint, reduce_lr])


cp_save_path = "/cephyr/NOBACKUP/groups/snic2021-23-496/scripts/All_data1_model.h5"
model.save(cp_save_path)

def import_data(directory, data_arr, IMG_CHANNELS):
    for root, subdirectories, files in sorted(os.walk(directory)):
    #print(root)
        for subdirectory in subdirectories:
            file_path = os.path.join(root, subdirectory)
            #print(subdirectory)
            for f in os.listdir(file_path):
                if f.endswith('.png'):
                    #print(f)
                    img_path=file_path + '/' + f   #create first of dic values, i.e the path
                    #print(img_path)
                    #print(img_path)
                    #imagename=ntpath.basename(imagepath)#take the name of the file from the path and save it
                    try:
                        img = imread(img_path)[:,:,:IMG_CHANNELS]
                        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
                    except IndexError:
                        pass
                    #X_train[n1] = img  #Fill empty X_train with values from img
                    data_arr.append(img)
    return(data_arr)
np_path = '/cephyr/NOBACKUP/groups/snic2021-23-496/np_data/'
X_train_k = np.load(np_path + 'X_Dataset_k_s512.npy')
Y_train_k = np.load(np_path + 'Y_Dataset_k_s512.npy')

val_X_train = np.load(np_path + 'Val_X_Dataset_k_s512.npy')
own_img_dirs = ['/home/inf-54-2020/experimental_cop/Train_H_Final/Aug_Img/OneThird/', '/home/inf-54-2020/experimental_cop/Train_H_Final/Aug_Img/TwoThird/', '/home/inf-54-2020/experimental_cop/Train_H_Final/Aug_Img/ThirdThird/', X_train_k]
own_masks_dirs = ['/home/inf-54-2020/experimental_cop/Train_H_Final/Aug_Mask/OneThird/', '/home/inf-54-2020/experimental_cop/Train_H_Final/Aug_Mask/TwoThird/', '/home/inf-54-2020/experimental_cop/Train_H_Final/Aug_Mask/ThirdThird/', Y_train_k]


def generate_arrays_from_file(sub_dir_no, own_img_dirs,own_masks_dirs):
    sub_dir_no = sub_dir_no -1
    empty_array = []
    while True:
        for i in range(sub_dir_no):
        #open the main dir e.g. OneThird (contains 1/3rd of all subdirs of train data)
            subd_x = own_img_dirs[i]
            subd_y = own_masks_dirs[i]
            X_train = import_data(subd_x, empty_array, 3)
            Y_train = import_data(subd_y, empty_array, 1)
            xy_arr = list(X_train, Y_train)
            yield(xy_arr)
        xy_arr_kagl = list(own_img_dirs[sub_dir_no],own_masks_dirs[sub_dir_no])
        yield(xy_arr_kagl)

checkpointer = tf.keras.callbacks.ModelCheckpoint(cp_save_path, verbose=1, save_best_only=True)
# def scheduler(epoch, lr): #keeps the initial learning rate (e.g. 0.01) for the first 5 epocsh and
#                             #then decreases it significantly
#    if epoch < 5:
#     return lr
#    else:
#      return lr * tf.math.exp(-0.1)
# initialize the number of epochs to train for and batch size

# Load data
def generate_arrays_from_file(imgs_masks_paths, masks_path, batchsize):
    inputs = []
    targets = []
    batchcount = 0
    rootdir = imgs_masks_paths
    for file in os.listdir(rootdir):

    while True:
        X_train = import_data(, empty_X, 3))

        with open(path) as f:
            for line in f:
                x,y = line.split(',')
                inputs.append(x)
                targets.append(y)
                batchcount += 1
                if batchcount > batchsize:
                  X_train = np.array(inputs, dtype='float32')
                  Y_train = np.array(targets, dtype='float32')
                  yield (X_train, Y_train)
                  inputs = []
                  targets = []
                  batchcount = 0

NUM_EPOCHS = 75
BS = 32
NUM_TRAIN_IMAGES = 0
NUM_TEST_IMAGES = 0

callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='logs'),
        #tf.keras.callbacks.LearningRateScheduler(scheduler)
]
model.fit_generator(generate_arrays_from_file(4, own_img_dirs,own_masks_dirs),
        validation_data=(X_train_k, Y_train_k), samples_per_epoch=200, nb_epoch=10)
model.fit_generator(generate_arrays_from_file(4, own_img_dirs,own_masks_dirs),
        validation_data=(X_train_k, Y_train_k), steps_per_epoch=NUM_TRAIN_IMAGES // BS, validation_steps=NUM_TEST_IMAGES // BS,
epochs=NUM_EPOCHS)



#results= model_segm.fit(np.array(X_train_list), np.array(Y_train_list), validation_split=0.1, batch_size=16, epochs=50, callbacks=callbacks)

# this generator loads data from the given directory and 32 images 
# chunks called batches. you can set this as you like
# train_generator = train_datagen.flow_from_directory(
#         dir_path,
#         target_size=(128, 128),
#         batch_size=32,
#         class_mode='binary')

#         #print(subdirectory)
# # same es the train_generator    
# validation_generator = test_datagen.flow_from_directory(
#         'data/validation',
#         target_size=(150, 150),
#         batch_size=32,
#         class_mode='binary')

# # loads sequentially images and feeds them to the model. 
# # the batch size is set in the constructor 
# model_segm.fit_generator(
#         train_generator,
#         samples_per_epoch=2000,
#         nb_epoch=50,
#         validation_data=validation_generator,
#         nb_val_samples=800)
# sdn = 3
# def dummy(sdn):
#     while 1:
#         for i in range(sdn):
#             #print(i)
#             yield(i)
#         final = sdn + 1
#         yield(final)
    
# for a in range(3+1):
#     print(a)
#     b = dummy(a)
#     print(b)