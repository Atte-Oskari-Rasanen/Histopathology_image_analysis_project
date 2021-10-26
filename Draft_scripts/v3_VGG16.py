#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 15:17:21 2021

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

def main():
    TRAIN_IMG_DIR = "/home/inf-54-2020/experimental_cop/Train_H_Final/Aug_Img/"
    M_TRAIN_IMG_DIR = "/home/inf-54-2020/experimental_cop/Train_H_Final/Aug_Mask/"
    
    #TRAIN_IMG_DIR = '/home/atte/kansio/img/'
    #M_TRAIN_IMG_DIR ='/home/atte/kansio/img_mask/'
    
    VAL_IMG_DIR = "/home/inf-54-2020/experimental_cop/Val_H_Final/Orginal_unpatched/"
    
    #Resizing images is optional, CNNs are ok with large images
    IMG_HEIGHT = 1024 #Resize images (height  = X, width = Y)
    IMG_WIDTH = 996
    IMG_CHANNELS = 3
    X_train = []
    Y_train = []
    n1 = 0
    n2 = 0
    print('starting the loops...')
    
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
    
    np.save('/home/inf-54-2020/experimental_cop/scripts/X_train_1024x996.npy', X_train)
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
       
    Y_train=np.array(Y_train)
    
    print(Y_train.shape)
    print('lengths of X_ train and Y_Train: ')
    print(len(X_train))
    print(len(Y_train))
    
    print('masks saved into array!')
    np.save('/home/inf-54-2020/experimental_cop/scripts/Y_train_1024x996.npy', Y_train)
    
    #X_train = np.load('/home/inf-54-2020/experimental_cop/scripts/X_train_size100.npy')
    #Y_train = np.load('/home/inf-54-2020/experimental_cop/scripts/Y_train_size100.npy')
    
    #Use customary x_train and y_train variables
    X_train = X_train
    Y_train = Y_train
    Y_train = np.expand_dims(Y_train, axis=3) #May not be necessary.. leftover from previous code 
    
    
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
    #New model with only first 2 conv layers
    new_model = Model(inputs=VGG_model.input, outputs=VGG_model.get_layer('block1_conv2').output)
    new_model.summary()
    
    #Now, let us apply feature extractor to our training data
    features=new_model.predict(X_train)
    
    #Plot features to view them
    square = 8
    ix=1
    # for _ in range(square):
    #     for _ in range(square):
    #         ax = plt.subplot(square, square, ix)
    #         ax.set_xticks([])
    #         ax.set_yticks([])
    #         plt.imshow(features[0,:,:,ix-1], cmap='gray')
    #         plt.savefig()
    #         ix +=1
    # plt.show()
    
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
    print('dataset look:')
    print(dataset)
    print('###########')
    print(dataset['Label'].unique())
    print(dataset['Label'].value_counts())
    
    ##If we do not want to include pixels with value 0 
    ##e.g. Sometimes unlabeled pixels may be given a value 0.
    #dataset = dataset[dataset['Label'] != 0]
    
    #Redefine X and Y for Random Forest
    X_for_RF = dataset.drop(labels = ['Label'], axis=1)
    Y_for_RF = dataset['Label']
    
    print('X and Y redefined for RF')
    #print(X_for_RF.dtypes)
    #print(Y_for_RF.dtypes)
    
    X_for_RF = X_for_RF.astype('int32')
    Y_for_RF = X_for_RF.astype('int32')
    
    ##################
    #RANDOM FOREST
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators = 50, random_state = 42)
    
    # Train the model on training data
    model.fit(X_for_RF, Y_for_RF) 
    
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
if __name__== '__main__':
    main()