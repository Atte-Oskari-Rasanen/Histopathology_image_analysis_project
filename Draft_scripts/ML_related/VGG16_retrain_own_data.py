#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 11:04:27 2021

@author: atte
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import glob
import cv2
import pickle

from keras.models import Sequential, Model
from keras.layers import Conv2D
import os
from keras.applications.vgg16 import VGG16
from matplotlib import image
from matplotlib import pyplot



#Resizing images is optional, CNNs are ok with large images
IMG_HEIGHT = 3600 #Resize images (height  = X, width = Y)
IMG_WIDTH = 5760
IMG_CHANNELS = 3
TRAIN_IMG_DIR = "/home/inf-54-2020/experimental_cop/Train_H_Final/Full_Aug_Img/"
M_TRAIN_IMG_DIR = "/home/inf-54-2020/experimental_cop/Train_H_Final/Full_Aug_Mask/"

X_train = []
Y_train = []
#Capture training image info as a list
train_images = []

print('starting the loops...')

img_dir_id = [] #list of dir ids containing patches of the certain image
ind_im_ids = [] #create an empty list for the ids of the individual images found in the subdir
n1 = 0
for imagefile in os.listdir(TRAIN_IMG_DIR):  #to go through files in the specific directory
    print(imagefile)
    img_path=TRAIN_IMG_DIR + '/' + imagefile   #create first of dic values, i.e the path
    #imagename=ntpath.basename(imagepath)#take the name of the file from the path and save it
    img = image.imread(img_path)[:,:,:IMG_CHANNELS]
    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    #X_train[n1] = img  #Fill empty X_train with values from img
    X_train.append(img)
    #print(str(n1) + ' one loop of X_train done!')
    n1 += 1
   
X_train=np.array(X_train)
print(X_train.shape)
np.save('/home/inf-54-2020/experimental_cop/scripts/X_train_size512.npy', X_train)

print('Images saved into array!')
n2 = 0
for imagefile in os.listdir(M_TRAIN_IMG_DIR):  #to go through files in the specific directory
    #print(f)
    img_path=M_TRAIN_IMG_DIR + '/' + imagefile   #create first of dic values, i.e the path
    #print(img_path)
    #print(img_path)
    #imagename=ntpath.basename(imagepath)#take the name of the file from the path and save it
    img = image.imread(img_path)[:,:,:1]
    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    #X_train[n1] = img  #Fill empty X_train with values from img
    Y_train.append(img)
    #print(str(n1) + ' one loop of Y_train done!')
    n1 += 1

Y_train=np.array(Y_train)

np.save('/home/inf-54-2020/experimental_cop/scripts/Y_train_size512.npy', Y_train)


#Use customary x_train and y_train variables
X_train = X_train
y_train = Y_train
y_train = np.expand_dims(y_train, axis=3) #May not be necessary.. leftover from previous code 


#Load VGG16 model wothout classifier/fully connected layers
#Load imagenet weights that we are going to use as feature generators
VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

#Make loaded layers as non-trainable. This is important as we want to work with pre-trained weights
for layer in VGG_model.layers:
	layer.trainable = False
    
VGG_model.summary()  #Trainable parameters will be 0

#After the first 2 convolutional layers the image dimension changes. 
#So for easy comparison to Y (labels) let us only take first 2 conv layers
#and create a new model to extract features
#New model with only first 2 conv layers
new_model = Model(inputs=VGG_model.input, outputs=VGG_model.get_layer('block1_conv2').output)
new_model.summary()

cp_save_path = "/home/inf-54-2020/experimental_cop/scripts/Model_VGG16_retrained_512.h5"
new_model.save(cp_save_path)

#Now, let us apply feature extractor to our training data
features=new_model.predict(X_train)
