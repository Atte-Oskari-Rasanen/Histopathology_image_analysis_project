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
# from keras import train_generator, validation_generator, fit_generator, train_datagen, test_datagen
seed = 42
np.random.seed = seed

#you need two functions for X_train and Y_train. One for importing kaggle, one for importing own
#img dims need to be divisible by 32
print('Starting!')
#To fix the memory issue use train_on_batch. First need to get the numpy arrays. 
#For this make 3 sets (for images and masks): each set contains kaggle data + 1/3 
#of the own data, then the create a numpy array for the rest (2/3rds). 
IMG_WIDTH = 512  #these dims needed IF you are to transform the kaggle arrays 
#back to images but in different dir format (needed for fit_generator while training a model)
IMG_HEIGHT = 512
IMG_CHANNELS = 2
# cp_save_path = '/home/inf-54-2020/experimental_cop/scripts/working_models/kaggle_model'
# model_segm = keras.models.load_model(cp_save_path)
kaggle_dir = '/home/inf-54-2020/experimental_cop/kaggle_data/'

# kaggle_dir = sys.argv[1]

# i_13 = sys.argv[2]
# i_23 = sys.argv[3]
# i_33 = sys.argv[4]
# m_13 = sys.argv[5]
# m_23 = sys.argv[6]
# m_33 = sys.argv[7]
val_kaggle_dir = '/home/inf-54-2020/experimental_cop/kaggle_data_val/'

# # own_img_dirs = [i_13, i_23, i_33]
# # own_masks_dirs = [m_13, m_23, m_33]

own_img_dirs = ['/home/inf-54-2020/experimental_cop/Train_H_Final/Aug_Img/OneThird/', '/home/inf-54-2020/experimental_cop/Train_H_Final/Aug_Img/TwoThird/', '/home/inf-54-2020/experimental_cop/Train_H_Final/Aug_Img/ThirdThird/']
own_masks_dirs = ['/home/inf-54-2020/experimental_cop/Train_H_Final/Aug_Mask/OneThird/', '/home/inf-54-2020/experimental_cop/Train_H_Final/Aug_Mask/TwoThird/', '/home/inf-54-2020/experimental_cop/Train_H_Final/Aug_Mask/ThirdThird/']
# val_own_data_dir = 'kaggle_data_val'


train_ids = next(os.walk(kaggle_dir))[1]

X_train =[]
Y_train = []

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):   
    path = kaggle_dir + id_
    #img = imread(path + '/images/' + id_ + '.png')[:,:,:2]  
    img = imread(path + '/images/' + id_ + '.png')  
    print(img.shape)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    print(img.shape)
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train.append(img)  #Fill empty X_train with values from img
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',  
                                      preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)  
            
    Y_train.append(mask)   

np_path = '/home/inf-54-2020/experimental_cop/scripts/np_data/'
np.save(np_path +'X_Dataset_k_s128.npy', X_train)
np.save(np_path + 'Y_Dataset_k_s128.npy', Y_train)

# train_ids = next(os.walk(val_kaggle_dir))[1]

# for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):   
#     path = val_kaggle_dir + id_
#     img = imread(path + '/images/' + id_ + '.png')  
#     print(img.shape)
#     img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
#     print(img.shape)
#     img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
#     X_train.append(img)  #Fill empty X_train with values from img
#     mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
#     for mask_file in next(os.walk(path + '/masks/'))[2]:
#         mask_ = imread(path + '/masks/' + mask_file)
#         mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',  
#                                       preserve_range=True), axis=-1)
#         mask = np.maximum(mask, mask_)  
            
#     Y_train.append(mask)   
# np_path = '/home/inf-54-2020/experimental_cop/scripts/np_data/'
# np.save(np_path +'val_X_Dataset_k_s128.npy', X_train)
# np.save(np_path + 'val_Y_Dataset_k_s128.npy', Y_train)


# # print('kaggle data saved!')
# def import_data(directory, data_arr, IMG_CHANNELS):
#     for root, subdirectories, files in sorted(os.walk(directory)):
#     #print(root)
#         for subdirectory in subdirectories:
#             file_path = os.path.join(root, subdirectory)
#             #print(subdirectory)
#             for f in os.listdir(file_path):
#                 if f.endswith('.png'):
#                     #print(f)
#                     img_path=file_path + '/' + f   #create first of dic values, i.e the path
#                     #print(img_path)
#                     #print(img_path)
#                     #imagename=ntpath.basename(imagepath)#take the name of the file from the path and save it
#                     try:
#                         img = imread(img_path)[:,:,:IMG_CHANNELS]
#                         img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
#                     except IndexError:
#                         pass
#                     #X_train[n1] = img  #Fill empty X_train with values from img
#                     data_arr.append(img)
#     return(data_arr)

# own_imgs_part_a = own_img_dirs[0]
# own_imgs_part_b = own_img_dirs[1]
# own_imgs_part_c = own_img_dirs[2]

# own_masks_part_a = own_masks_dirs[0]
# own_masks_part_b = own_masks_dirs[1]
# own_masks_part_c = own_masks_dirs[2]

# empty_X = []
# empty_Y = []
# X_train_a = import_data(own_imgs_part_a, empty_X, 1)
# Y_train_a = import_data(own_masks_part_a, empty_Y, 1)
# np.save('/home/inf-54-2020/experimental_cop/scripts/np_data/BW_X_Dataset_a_s512.npy', X_train_a)
# np.save('/home/inf-54-2020/experimental_cop/scripts/np_data/BW_Y_Dataset_a_s512.npy', Y_train_a)

# # print('dataset a saved!')

# X_train_b = import_data(own_imgs_part_b, empty_X, 1)
# Y_train_b = import_data(own_masks_part_b, empty_Y, 1)
# np.save('/home/inf-54-2020/experimental_cop/scripts/np_data/BW_X_Dataset_b_s512.npy', X_train_b)
# np.save('/home/inf-54-2020/experimental_cop/scripts/np_data/BW_Y_Dataset_b_s512.npy', Y_train_b)

# print('dataset b saved!')

# X_train_c = import_data(own_imgs_part_c, empty_X, 1)
# Y_train_c = import_data(own_masks_part_c, empty_Y, 1)
# np.save('/home/inf-54-2020/experimental_cop/scripts/np_data/BW_X_Dataset_c_s512.npy', X_train_c)
# np.save('/home/inf-54-2020/experimental_cop/scripts/np_data/BW_Y_Dataset_c_s512.npy', Y_train_c)
# print('dataset c saved!')



