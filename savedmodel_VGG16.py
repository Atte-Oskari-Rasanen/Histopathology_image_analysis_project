#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 17:57:16 2021

@author: atte
"""
import tensorflow as tf
import os
import random
import numpy as np
 
from tqdm import tqdm 
import pickle

from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import re

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

IMG_HEIGHT = 128 #Resize images (height  = X, width = Y)
IMG_WIDTH = 128
IMG_CHANNELS = 3
cp_save_path = "/home/inf-54-2020/experimental_cop/scripts/kaggle_model.h5"

X_train = np.load('/home/inf-54-2020/experimental_cop/scripts/kd_X_train_size128.npy')
Y_train = np.load('/home/inf-54-2020/experimental_cop/scripts/kd_Y_train_size128.npy')

Y_train = np.expand_dims(Y_train, axis=3) #May not be necessary.. leftover from previous code 

print('Starting...')
#Load VGG16 model wothout classifier/fully connected layers
#Load imagenet weights that we are going to use as feature generators
VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

#Make loaded layers as non-trainable. This is important as we want to work with pre-trained weights
for layer in VGG_model.layers:
	layer.trainable = False
    
VGG_model.summary()  #Trainable parameters will be 0

#After the first 2 convolutional layers the image dimension changes. 
#So for easy comparison to Y (labels) let us only take first 2 conv layers
#and create a new model to extract features
#New model with only first 2 conv layers -- this is our feature extractor!
new_model = Model(inputs=VGG_model.input, outputs=VGG_model.get_layer('block1_conv2').output)
new_model.summary()
print('prediction run')

#Now, let us apply feature extractor to our training data
features=new_model.predict(X_train)
#Plot features to view them
square = 8
ix=1
# =============================================================================
# for _ in range(square):
#     for _ in range(square):
#         ax = plt.subplot(square, square, ix)
#         ax.set_xticks([])
#         ax.set_yticks([])
#         plt.imshow(features[0,:,:,ix-1], cmap='gray')
#         ix +=1
# plt.show()
# plt.savefig('/home/inf-54-2020/experimental_cop/scripts/features'+str(ix)+'.png')
# =============================================================================


#Reassign 'features' as X to make it easy to follow
X=features

print('X shape')
print(X.shape)
X = X.reshape(-1, X.shape[3])  #Make it compatible for Random Forest by collapsing
#the dims into a single column and match Y labels
print('reshaped X')
print(X.shape)

print('X_train, Y_train shape')
print(X_train.shape)
print(Y_train.shape)
#Reshape Y to match X
Y_train = np.squeeze(Y_train) #remove dims
Y = Y_train.reshape(-1)
print('reshaped Y')
print(Y.shape)
#Combine X and Y into a dataframe to make it easy to drop all rows with Y values 0
#In our labels Y values 0 = unlabeled pixels. 
dataset = pd.DataFrame(X)
print('df prior to adding Y')
#print(dataset)
#print(dataset.info)
dataset['Label'] = Y
print('print the label column')
print(dataset['Label'])
#print(dataset['Label'].value_counts())

##If we do not want to include pixels with value 0 
##e.g. Sometimes unlabeled pixels may be given a value 0.
#dataset = dataset[dataset['Label'] != 0]
print('datasets done')
#Redefine X and Y for Random Forest
X_for_RF = dataset.drop(labels = ['Label'], axis=1)
Y_for_RF = dataset['Label']

print('prepping X and Y trains for the Rf...')
#print(X_for_RF.dtypes)
#print(Y_for_RF.dtypes)

X_for_RF = X_for_RF.astype('int32')
Y_for_RF = X_for_RF.astype('int32')
print('Starting up random forest...')
#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 50, random_state = 42)
print('train the model on the data')
# Train the model on training data
model.fit(X_for_RF, Y_for_RF) 

#Save model for future use

cp_save_path = "/home/inf-54-2020/experimental_cop/scripts/RF_model.h5"
model.save(cp_save_path)

#Save model for future use

#Load model.... 

#Test on a different image
#READ EXTERNAL IMAGE...
test_img = cv2.imread('/home/inf-54-2020/experimental_cop/test.png', cv2.IMREAD_COLOR)       
test_img = cv2.resize(test_img, (IMG_HEIGHT, IMG_WIDTH))
test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
test_img = np.expand_dims(test_img, axis=0)

#predict_image = np.expand_dims(X_train[8,:,:,:], axis=0)
X_test_feature = new_model.predict(test_img)
X_test_feature = X_test_feature.reshape(-1, X_test_feature.shape[3])

prediction = model.predict(X_test_feature)

#View and Save segmented image
prediction_image = prediction.reshape(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS)
plt.imshow(prediction_image, cmap='gray')
plt.imsave('/home/inf-54-2020/experimental_cop/360_segmented.jpg', prediction_image, cmap='gray')
print('All done!')