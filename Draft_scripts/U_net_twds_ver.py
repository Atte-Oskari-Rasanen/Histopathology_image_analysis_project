#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 18:37:34 2021

@author: atte
"""

import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
import pandas as pd
# from segmentation_models import Unet
# from segmentation_models.backbones import get_preprocessing
# from segmentation_models.losses import bce_jaccard_loss
# from segmentation_models.metrics import iou_score
# from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from keras.models import model_from_json



import cv2
import numpy as np
from keras.layers import Input, Conv2D, Reshape
from keras.models import Model
import tensorflow as tf
import os
import random
import numpy as np
 
from tqdm import tqdm 

from skimage.io import imread, imshow
from skimage.transform import resize

seed = 42
np.random.seed = seed

IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 3

#TRAIN_PATH = '/cephyr/NOBACKUP/groups/snic2021-23-496/kaggle_data/'
TRAIN_PATH = '/home/inf-54-2020/experimental_cop/kaggle_data/'
#TEST_PATH = 'stage1_test/'
#X_test = np.load('/home/inf-54-2020/experimental_cop/scripts/X_test_size128.npy')

train_ids = next(os.walk(TRAIN_PATH))[1]
#test_ids = next(os.walk(TEST_PATH))[1]

# X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
# Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

# print('Resizing training images and masks')
# for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):   
#     path = TRAIN_PATH + id_
#     img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]  
#     img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
#     X_train[n] = img  #Fill empty X_train with values from img
#     mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
#     for mask_file in next(os.walk(path + '/masks/'))[2]:
#         mask_ = imread(path + '/masks/' + mask_file)
#         mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',  
#                                       preserve_range=True), axis=-1)
#         mask = np.maximum(mask, mask_)  
            
#     Y_train[n] = mask   
# np.save('/home/inf-54-2020/experimental_cop/scripts/X_train_size512.npy', X_train)
# np.save('/home/inf-54-2020/experimental_cop/scripts/Y_train_size512.npy', Y_train)

X_train = np.load('/home/inf-54-2020/experimental_cop/scripts/X_train_size512.npy')
Y_train = np.load('/home/inf-54-2020/experimental_cop/scripts/Y_train_size512.npy')

print(X_train.shape)
print(Y_train.shape)
from tensorflow.keras.layers import UpSampling2D


#build the model
def get_model():
    IMG_WIDTH = 512
    IMG_HEIGHT = 512
    IMG_CHANNELS = 3

    in1 = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(in1)
    conv1 = tf.keras.layers.Dropout(0.2)(conv1)
    conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D((2, 2))(conv1)

    conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool1)
    conv2 = tf.keras.layers.Dropout(0.2)(conv2)
    conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D((2, 2))(conv2)

    conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool2)
    conv3 = tf.keras.layers.Dropout(0.2)(conv3)
    conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv3)
    pool3 = tf.keras.layers.MaxPooling2D((2, 2))(conv3)

    conv4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool3)
    conv4 = tf.keras.layers.Dropout(0.2)(conv4)
    conv4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv4)

    up1 = tf.keras.layers.concatenate([UpSampling2D((2, 2))(conv4), conv3], axis=-1)
    conv5 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(up1)
    conv5 = tf.keras.layers.Dropout(0.2)(conv5)
    conv5 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv5)
    
    up2 = tf.keras.layers.concatenate([UpSampling2D((2, 2))(conv5), conv2], axis=-1)
    conv6 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(up2)
    conv6 = tf.keras.layers.Dropout(0.2)(conv6)
    conv6 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv6)

    up2 = tf.keras.layers.concatenate([UpSampling2D((2, 2))(conv6), conv1], axis=-1)
    conv7 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(up2)
    conv7 = tf.keras.layers.Dropout(0.2)(conv7)
    conv7 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv7)
    segmentation = tf.keras.layers.Conv2D(3, (1, 1), activation='sigmoid', name='seg')(conv7)

    model = tf.keras.Model(inputs=[in1], outputs=[segmentation])

    losses = {'seg': 'binary_crossentropy'
            }

    metrics = {'seg': ['acc']
                }
    model.compile(optimizer="adam", loss = losses, metrics=metrics)

    return model

model_name = "kaggle_model512.h5"

model = get_model()
model.summary()
cp_save_path = "/home/inf-54-2020/experimental_cop/scripts/kaggle_model_size512.h5"

checkpointer = tf.keras.callbacks.ModelCheckpoint(cp_save_path, verbose=1, save_best_only=True)

callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='logs')]


model.save(cp_save_path)
checkpointer = tf.keras.callbacks.ModelCheckpoint(cp_save_path, verbose=1, save_best_only=True)


print('Model built and saved, now fitting it...')
history = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=200, callbacks=callbacks)

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

#evaluate the performance on the test set:
model.evaluate(x_test, y_test)
