#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 19:08:46 2021

@author: atte
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 

import glob
import cv2
import pickle
from PIL import Image

from keras.models import Sequential, Model
from keras.layers import Conv2D
import os
from keras.applications.vgg16 import VGG16

#TRAIN_IMG_DIR = '/home/atte/Documents/images_qupath2/H_final/'
#M_TRAIN_IMG_DIR='/home/atte/Documents/images_qupath2/H_final/Masks/'

TRAIN_IMG_DIR = "/home/inf-54-2020/experimental_cop/Train_H_Final/Aug_Img/"
M_TRAIN_IMG_DIR = "/home/inf-54-2020/experimental_cop/Train_H_Final/Aug_Mask/"


VAL_IMG_DIR = "/home/inf-54-2020/experimental_cop/Val_H_Final/Images/"

#Resizing images is optional, CNNs are ok with large images
#SIZE_X = 224 #Resize images (height  = X, width = Y)
#SIZE_Y = 224

#Capture training image info as a list
train_images = []

#Get a list of sub dirs from the main dir:
#subdirs_T = os.listdir(TRAIN_IMG_DIR) 
p = '/home/atte/kansio/'
for root, subdirectories, files in os.walk(p):
    #print(root)
    print(subdirectories)
    #print(files)
    for subdirectory in subdirectories:
        #print(subdirectory)
        file_path = os.path.join(root, subdirectory)
        im_files = os.listdir(file_path)
        print(im_files)
        print(file_path)
        #for f in os.listdir(file_path):
            #print(f)
        #print(os.path.join(root, subdirectory))
        break

failed_files = []
for root, subdirectories, files in os.walk(TRAIN_IMG_DIR):
    for subdirectory in subdirectories:
        file_path = os.path.join(root, subdirectory)
        for f in os.listdir(file_path):
            img_path=file_path + f   #create first of dic values, i.e the path
                #print(imagepath)
            #imagename=ntpath.basename(imagepath)#take the name of the file from the path and save it
            img = Image.open(img_path)       
            print(img_path)
            #img = np.resize(3600, 5760) #resize not working
            img_arr = np.array(img)
            if img_arr.size <= 1: #to avoid a potential error. skip such a file
            #save the image name so that the corresponing mask can be skipped too 
                failed_files.append(f)
                continue
            #img_arr = np.asarray(img)
            #print(img_arr[1])
            #print(img_arr.shape)
            #print(img_arr.shape[1])
            
            #print(img_arr.shape)
            #img = cv2.resize(img, (SIZE_Y, SIZE_X))
            #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            train_images.append(img_arr)
    
train_img_path = TRAIN_IMG_DIR + 'train_images_array.npy'
train_images = np.array(train_images)
np.save(train_img_path, train_images) #save numpy array just in case
#Capture mask/label info as a list
train_masks = [] 
for imagefile in os.listdir(M_TRAIN_IMG_DIR):
    if imagefile.endswith('.png'): #exclude files not ending in .tif
        if imagefile in failed_files:
            continue
        mask_path=M_TRAIN_IMG_DIR + imagefile   #create first of dic values, i.e the path
    #for directory_path in glob.glob("/home/inf-54-2020/experimental_cop/H_final/Masks/"):
        mask = Image.open(mask_path)
    #    for mask_path in glob.glob(os.path.join(directory_path, "*.png")):
        mask_arr = np.array(mask)
        if mask_arr.size <= 1: #to avoid a potential error. skip such a file
        #save the image name so that the corresponing mask can be skipped too 
            continue
        #img_arr = np.asarray        #mask = cv2.resize(mask, (SIZE_Y, SIZE_X))
        #mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
        mask_patches = make_patches(mask_arr)
        train_masks.append(mask_patches)
    #train_labels.append(label)
#Convert list to array for machine learning processing          
train_mask_path = M_TRAIN_IMG_DIR + 'train_masks_array.npy'
train_masks = np.array(train_masks)
np.save(train_mask_path, train_masks) #save numpy array just in case


print("images and masks saved to arrays!")
#Use customary x_train and y_train variables
X_train = train_images
y_train = train_masks
y_train = np.expand_dims(y_train, axis=3) #May not be necessary.. leftover from previous code 


#Load VGG16 model wothout classifier/fully connected layers
#Load imagenet weights that we are going to use as feature generators
#include_toip=false means that dont ouput the dense layers
VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(600, 960, 3))
print('loaded the VGG model!')
#Make loaded layers as non-trainable since we want to work with pre-trained weights!!!
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

# =============================================================================
# #Plot features to view them
# square = 8
# ix=1
# for _ in range(square):
#     for _ in range(square):
#         ax = plt.subplot(square, square, ix)
#         ax.set_xticks([])
#         ax.set_yticks([])
#         plt.imshow(features[0,:,:,ix-1], cmap='gray')
#         ix +=1
# plt.show()
# plt.savefig('Plot1.pdf')
# 
# =============================================================================

#Reassign 'features' as X to make it easy to follow
X=features
X = X.reshape(-1, X.shape[3])  #Make it compatible for Random Forest and match Y labels

#Reshape Y to match X
Y = y_train.reshape(-1)

#Combine X and Y into a dataframe to make it easy to drop all rows with Y values 0
#In our labels Y values 0 = unlabeled pixels. 
dataset = pd.DataFrame(X)
dataset['Label'] = Y
print(dataset['Label'].unique())
print(dataset['Label'].value_counts())

##If we do not want to include pixels with value 0 
##e.g. Sometimes unlabeled pixels may be given a value 0.
dataset = dataset[dataset['Label'] != 0]

#Redefine X and Y for Random Forest
X_for_RF = dataset.drop(labels = ['Label'], axis=1)
Y_for_RF = dataset['Label']

#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 50, random_state = 42)

# Train the model on training data
model.fit(X_for_RF, Y_for_RF) 

#Save model for future use
filename = 'H_detector_model.pickle'
pickle.dump(model, open(filename, 'wb'))

# =============================================================================
# #Load model.... 
# loaded_model = pickle.load(open(filename, 'rb'))
# 
# #Test on a different image
# #READ EXTERNAL IMAGE...
# test_img = cv2.imread('/home/inf-54-2020/experimental_cop/Val_H_Final/Images/YZ004_NR_G2_#15_hCOL1A1_20x_1_H_Final.tif', cv2.IMREAD_COLOR)       
# test_img = cv2.resize(test_img, (SIZE_Y, SIZE_X))
# test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
# test_img = np.expand_dims(test_img, axis=0)
# 
# #predict_image = np.expand_dims(X_train[8,:,:,:], axis=0)
# X_test_feature = new_model.predict(test_img)
# X_test_feature = X_test_feature.reshape(-1, X_test_feature.shape[3])
# 
# prediction = loaded_model.predict(X_test_feature)
# 
# #View and Save segmented image
# prediction_image = prediction.reshape(mask.shape)
# plt.imshow(prediction_image, cmap='gray')
# plt.imsave('/home/inf-54-2020/experimental_cop/Val_H_Final/Output/YZ004_NR_G2_#15_hCOL1A1_20x_1_H_Final.tif', prediction_image, cmap='gray')
# 
# =============================================================================
