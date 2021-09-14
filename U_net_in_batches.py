#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 14:18:45 2021

@author: atte
"""
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

#you need two functions for X_train and Y_train. One for importing kaggle, one for importing own
#img dims need to be divisible by 32
IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 3
cp_save_path = '/home/inf-54-2020/experimental_cop/scripts/working_models/kaggle_model'
model_segm = keras.models.load_model(cp_save_path)

own_img_dirs = ['/home/inf-54-2020/experimental_cop/Train_H_Final/Aug_Img/OneThird/', '/home/inf-54-2020/experimental_cop/Train_H_Final/Aug_Img/TwoThird/', '/home/inf-54-2020/experimental_cop/Train_H_Final/Aug_Img/ThirdThird/']
own_masks_dirs = ['/home/inf-54-2020/experimental_cop/Train_H_Final/Aug_Mask/OneThird/', '/home/inf-54-2020/experimental_cop/Train_H_Final/Aug_Mask/TwoThird/', '/home/inf-54-2020/experimental_cop/Train_H_Final/Aug_Mask/ThirdThird/']
own_imgs_part_a = own_img_dirs[0]
own_imgs_part_b = own_img_dirs[1]
own_imgs_part_b = own_img_dirs[2]

own_masks_part_a = own_masks_dirs[0]
own_masks_part_b = own_masks_dirs[1]
own_masks_part_b = own_masks_dirs[2]


X_train_a = np.load('/home/inf-54-2020/experimental_cop/scripts/np_datas/X_Dataset_a_s512.npy')
Y_train_a = np.load('/home/inf-54-2020/experimental_cop/scripts/np_datas/Y_Dataset_a_s512.npy')
X_train_b = np.save('/home/inf-54-2020/experimental_cop/scripts/np_datas/X_Dataset_b_s512.npy')
Y_train_b = np.save('/home/inf-54-2020/experimental_cop/scripts/np_datas/Y_Dataset_b_s512.npy')
X_train_c = np.load('/home/inf-54-2020/experimental_cop/scripts/np_datas/X_Dataset_c_s512.npy')
Y_train_c = np.load('/home/inf-54-2020/experimental_cop/scripts/np_datas/Y_Dataset_c_s512.npy')

X_train_list = [X_train_a, X_train_b, X_train_c]
Y_train_list = [Y_train_a, Y_train_a, Y_train_c]

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

results= model_segm.fit(np.array(X_train_list), np.array(Y_train_list), validation_split=0.1, batch_size=16, epochs=50, callbacks=callbacks)

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
