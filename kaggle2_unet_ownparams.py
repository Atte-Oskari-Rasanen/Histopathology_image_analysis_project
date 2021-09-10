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

IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 3

TRAIN_PATH = '/cephyr/NOBACKUP/groups/snic2021-23-496/kaggle_data/'
TRAIN_PATH = '/home/inf-54-2020/experimental_cop/kaggle_data/'
TRAIN_PATH = sys.argv[1]

TEST_PATH = 'stage1_test/'
train_ids = next(os.walk(TRAIN_PATH))[1]
#test_ids = next(os.walk(TEST_PATH))[1]

# X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
# Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

X_train = []
Y_train = []
print('Resizing training images and masks')
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):   
    path = TRAIN_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]  
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train.append(img)  #Fill empty X_train with values from img
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',  
                                      preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)  
            
    Y_train.append(mask)   

# np.save('/home/inf-54-2020/experimental_cop/scripts/X_train_size512.npy', X_train)
# np.save('/home/inf-54-2020/experimental_cop/scripts/Y_train_size512.npy', Y_train)
TRAIN_IMG_DIR = sys.argv[2]
M_TRAIN_IMG_DIR = sys.argv[3]

img_dir_id = [] #list of dir ids containing patches of the certain image
ind_im_ids = [] #create an empty list for the ids of the individual images found in the subdir
n1 = 0
for root, subdirectories, files in sorted(os.walk(TRAIN_IMG_DIR)):
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
                img = imread(img_path)[:,:,:IMG_CHANNELS]
                img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
                #X_train[n1] = img  #Fill empty X_train with values from img
                X_train.append(img)
                #print(str(n1) + ' one loop of X_train done!')
                n1 += 1
   
X_train=np.array(X_train)
#np.save('/home/inf-54-2020/experimental_cop/scripts/X_train_size128.npy', X_train)

print('Images saved into array!')
n2 = 0
for root, subdirectories, files in sorted(os.walk(M_TRAIN_IMG_DIR)):
    #print(root)
    for subdirectory in subdirectories:
        file_path = os.path.join(root, subdirectory)
        #print(subdirectory)
        for m in os.listdir(file_path):
            if m.endswith('.png'):
                #print(f)
                img_path=file_path + '/' + m   #create first of dic values, i.e the path
                #print(img_path)
                #print(img_path)
                #imagename=ntpath.basename(imagepath)#take the name of the file from the path and save it
                img = imread(img_path)[:,:,:1]
                img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
                #X_train[n1] = img  #Fill empty X_train with values from img
                Y_train.append(img)
                #print(str(n1) + ' one loop of Y_train done!')
                n1 += 1
            else:
                continue
Y_train=np.array(Y_train)

np.save('/cephyr/NOBACKUP/groups/snic2021-23-496/X_train_kagl_own_s512.npy', X_train)
np.save('/cephyr/NOBACKUP/groups/snic2021-23-496/Y_train_kagl_own_s512.npy', Y_train)

X_train = np.load('/cephyr/NOBACKUP/groups/snic2021-23-496/scripts/X_train_kagl_own_s512.npy')
Y_train = np.load('/cephyr/NOBACKUP/groups/snic2021-23-496/scripts/X_train_kagl_own_s512.npy')


def build_model(input_layer, start_neurons):
    #s = tf.keras.layers.Lambda(lambda x: x / 255)(input_layer) #normalise

    conv1 = tf.keras.layers.Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(input_layer)
    conv1 = tf.keras.layers.Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)
    pool1 = tf.keras.layers.MaxPooling2D((2, 2))(conv1)
    pool1 = tf.keras.layers.Dropout(0.25)(pool1)

    conv2 = tf.keras.layers.Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = tf.keras.layers.Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = tf.keras.layers.MaxPooling2D((2, 2))(conv2)
    pool2 = tf.keras.layers.Dropout(0.5)(pool2)

    conv3 = tf.keras.layers.Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = tf.keras.layers.Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(conv3)
    pool3 = tf.keras.layers.MaxPooling2D((2, 2))(conv3)
    pool3 = tf.keras.layers.Dropout(0.5)(pool3)

    conv4 = tf.keras.layers.Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(pool3)
    conv4 = tf.keras.layers.Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(conv4)
    pool4 = tf.keras.layers.MaxPooling2D((2, 2))(conv4)
    pool4 = tf.keras.layers.Dropout(0.5)(pool4)

    # Middle
    convm = tf.keras.layers.Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(pool4)
    convm = tf.keras.layers.Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(convm)
    
    deconv4 = tf.keras.layers.Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = tf.keras.layers.concatenate([deconv4, conv4])
    uconv4 = tf.keras.layers.Dropout(0.5)(uconv4)
    uconv4 = tf.keras.layers.Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)
    uconv4 = tf.keras.layers.Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)

    deconv3 = tf.keras.layers.Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = tf.keras.layers.concatenate([deconv3, conv3])
    uconv3 = tf.keras.layers.Dropout(0.5)(uconv3)
    uconv3 = tf.keras.layers.Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)
    uconv3 = tf.keras.layers.Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)

    deconv2 = tf.keras.layers.Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = tf.keras.layers.concatenate([deconv2, conv2])
    uconv2 = tf.keras.layers.Dropout(0.5)(uconv2)
    uconv2 = tf.keras.layers.Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
    uconv2 = tf.keras.layers.Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)

    deconv1 = tf.keras.layers.Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = tf.keras.layers.concatenate([deconv1, conv1])
    uconv1 = tf.keras.layers.Dropout(0.5)(uconv1)
    uconv1 = tf.keras.layers.Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
    uconv1 = tf.keras.layers.Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
    
    output_layer = tf.keras.layers.Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)

    return output_layer

input_layer = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
output_layer = build_model(input_layer, 16)

model = tf.keras.Model(inputs=[input_layer], outputs=[output_layer])

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

cp_save_path = "/cephyr/NOBACKUP/groups/snic2021-23-496/scripts/kaggle_model_size512.h5"

checkpointer = tf.keras.callbacks.ModelCheckpoint(cp_save_path, verbose=1, save_best_only=True)

callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='logs')]


model.save(cp_save_path)
checkpointer = tf.keras.callbacks.ModelCheckpoint(cp_save_path, verbose=1, save_best_only=True)
#model.save_weights(cp_save_path)
print('Model built and saved, now fitting it...')
history = model.fit(X_train, Y_train, validation_split=0.1, batch_size=128, epochs=200, callbacks=callbacks)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('plot1_kaggledata.png')

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('plot2_kaggledata.png')


print('Done.')
####################################

