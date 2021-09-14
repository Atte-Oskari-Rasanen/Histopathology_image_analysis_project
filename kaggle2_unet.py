#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 13:38:03 2021

@author: atte
"""
#source: https://www.kaggle.com/phoenigs/u-net-dropout-augmentation-stratification 
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
seed = 42
np.random.seed = seed

#img dims need to be divisible by 32
IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 3

# TRAIN_PATH = sys.argv[1]

# # # # # TEST_PATH = 'stage1_test/'
# train_ids = next(os.walk(TRAIN_PATH))[1]
# # # #test_ids = next(os.walk(TEST_PATH))[1]

# # # # X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
# # # # Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

# X_train = []
# Y_train = []
# print('Resizing training images and masks')
# for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):   
#     path = TRAIN_PATH + id_
#     img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]  
#     img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
#     X_train.append(img)  #Fill empty X_train with values from img
#     mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
#     for mask_file in next(os.walk(path + '/masks/'))[2]:
#         mask_ = imread(path + '/masks/' + mask_file)
#         mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',  
#                                       preserve_range=True), axis=-1)
#         mask = np.maximum(mask, mask_)  
            
#     Y_train.append(mask)   

X_train = np.load('/cephyr/NOBACKUP/groups/snic2021-23-496/scripts/X_train_size512.npy')
Y_train = np.load('/cephyr/NOBACKUP/groups/snic2021-23-496/scripts/Y_train_size512.npy')

# TRAIN_IMG_DIR = sys.argv[1]
# M_TRAIN_IMG_DIR = sys.argv[2]

# # img_dir_id = [] #list of dir ids containing patches of the certain image
# # ind_im_ids = [] #create an empty list for the ids of the individual images found in the subdir
# n1 = 0
# for root, subdirectories, files in sorted(os.walk(TRAIN_IMG_DIR)):
#     #print(root)
#     for subdirectory in subdirectories:
#         file_path = os.path.join(root, subdirectory)
#         #print(subdirectory)
#         for f in os.listdir(file_path):
#             if f.endswith('.png'):
#                 #print(f)
#                 img_path=file_path + '/' + f   #create first of dic values, i.e the path
#                 #print(img_path)
#                 #print(img_path)
#                 #imagename=ntpath.basename(imagepath)#take the name of the file from the path and save it
#                 try:
#                     img = imread(img_path)[:,:,:IMG_CHANNELS]
#                     img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
#                 except IndexError:
#                     pass
#                 #X_train[n1] = img  #Fill empty X_train with values from img
#                 X_train.append(img)
#                 #print(str(n1) + ' one loop of X_train done!')
#                 n1 += 1
   
# X_train=np.array(X_train)
# #np.save('/home/inf-54-2020/experimental_cop/scripts/X_train_size128.npy', X_train)

# print('Images saved into array!')
# n2 = 0
# for root, subdirectories, files in sorted(os.walk(M_TRAIN_IMG_DIR)):
#     #print(root)
#     for subdirectory in subdirectories:
#         file_path = os.path.join(root, subdirectory)
#         #print(subdirectory)
#         for m in os.listdir(file_path):
#             if m.endswith('.png'):
#                 #print(f)
#                 img_path=file_path + '/' + m   #create first of dic values, i.e the path
#                 #print(img_path)
#                 #print(img_path)
#                 #imagename=ntpath.basename(imagepath)#take the name of the file from the path and save it
#                 try:
#                     img = imread(img_path)[:,:,:1]
#                     img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
#                 except IndexError:
#                     pass

#                 #X_train[n1] = img  #Fill empty X_train with values from img
#                 Y_train.append(img)
#                 #print(str(n1) + ' one loop of Y_train done!')
#                 n1 += 1
#             else:
#                 continue
# Y_train=np.array(Y_train)

# np.save('/cephyr/NOBACKUP/groups/snic2021-23-496/X_train_kagl_own_s512.npy', X_train)
# np.save('/cephyr/NOBACKUP/groups/snic2021-23-496/X_train_kagl_own_s512.npy', Y_train)
# np.save('/cephyr/NOBACKUP/groups/snic2021-23-496/X_train_own_s512.npy', X_train)
# np.save('/cephyr/NOBACKUP/groups/snic2021-23-496/Y_train_own_s512.npy', Y_train)

print('shapes and sizes of X_train and Y_train:')
print(X_train.shape)
print(len(X_train))
print('========')
print(Y_train.shape)
print(len(Y_train))


#Build the model
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


cp_save_path = "/cephyr/NOBACKUP/groups/snic2021-23-496/scripts/kaggle_model_size512_orig_sett.h5"
model.save(cp_save_path)

checkpointer = tf.keras.callbacks.ModelCheckpoint(cp_save_path, verbose=1, save_best_only=True)
# def scheduler(epoch, lr): #keeps the initial learning rate (e.g. 0.01) for the first 5 epocsh and
#                             #then decreases it significantly
#    if epoch < 5:
#     return lr
#    else:
#      return lr * tf.math.exp(-0.1)

callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='logs'),
        #tf.keras.callbacks.LearningRateScheduler(scheduler)
]
# lr_schedule = keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=1e-2,
#     decay_steps=10000,
#     decay_rate=0.9)

#gradient norm scaling
#optimizer = keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)
#gradient norm clipping
# optimizer = keras.optimizers.Adam(learning_rate=lr_schedule, clipvalue=1.5)
#optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
# optimizer = tf.keras.optimizers.RMSprop(
#     learning_rate=lr_schedule,
#     rho=0.9,
#     epsilon=1e-07,
#     centered=False,
#     name="RMSprop",
    
# )

#model.save_weights(cp_save_path)
print('Model built and saved, now fitting it...')
history = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=50, callbacks=callbacks)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('plot1_kaggle_s512_orig.png')

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('plot2_kaggle_s512_orig.png')


print('Done.')
####################################

