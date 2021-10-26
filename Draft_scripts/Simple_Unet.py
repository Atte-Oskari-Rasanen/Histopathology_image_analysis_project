#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 21:04:27 2021

@author: atte
"""

import tensorflow as tf
import segmentation_models as sm
import glob
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

import random
import numpy as np
 
from tqdm import tqdm 

from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import re
BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)


#Resizing images is optional, CNNs are ok with large images
IMG_HEIGHT = 256 #Resize images (height  = X, width = Y)
IMG_WIDTH = 256

TRAIN_IMG_DIR = "/home/inf-54-2020/experimental_cop/Train_H_Final/Aug_Img/"
M_TRAIN_IMG_DIR = "/home/inf-54-2020/experimental_cop/Train_H_Final/Aug_Mask/"
VAL_IMG_DIR = "/home/inf-54-2020/experimental_cop/Val_H_Final/Images/"

TRAIN_IMG_DIR = '/home/atte/kansio/img/'
M_TRAIN_IMG_DIR ='/home/atte/kansio/img_mask/'

#Capture training image info as a list
train_images = []
X = []
Y = []
#for directory_path in glob.glob(TRAIN_IMG_DIR):

for root, subdirectories, files in sorted(os.walk(TRAIN_IMG_DIR)): #tqdm shows the progress bar of the for loop
    #print(root)
    for subdirectory in subdirectories:
        print(subdirectory)
        file_path = os.path.join(root, subdirectory)
        for f in os.listdir(file_path):
            if f.endswith('.png'):
                print(f)
                img_path=file_path + '/' + f   #create first of dic values, i.e the path
                print(img_path)
                #print(img_path)
                #imagename=ntpath.basename(imagepath)#take the name of the file from the path and save it
                img = imread(img_path)#[:,:,:IMG_CHANNELS]
                img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
                #X_train[n1] = img  #Fill empty X_train with values from img
                X.append(img)
        #train_labels.append(label)
#Convert list to array for machine learning processing        

X = np.array(X)
print('X shape:')
print(X.shape)
#Capture mask/label info as a list
for root, subdirectories, files in sorted(os.walk(M_TRAIN_IMG_DIR)): #tqdm shows the progress bar of the for loop
    #print(root)
    for subdirectory in subdirectories:
        print(subdirectory)
        file_path = os.path.join(root, subdirectory)
        for m in os.listdir(file_path):
            if m.endswith('.png'):
                
                mask_path=file_path + '/' + m   #create first of dic values, i.e the path
                print(img_path)
                #print(img_path)
                #imagename=ntpath.basename(imagepath)#take the name of the file from the path and save it
                mask = imread(img_path)#[:,:,:IMG_CHANNELS]
                mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
                #X_train[n1] = img  #Fill empty X_train with values from img
                Y.append(mask)
        #train_labels.append(label)
#Convert list to array for machine learning processing        
Y = np.array(Y)

print('Y shape:')
print(Y.shape)

Z = []
#print(root)
for root, subdirectories, files in sorted(os.walk(VAL_IMG_DIR)): #tqdm shows the progress bar of the for loop

    for subdirectory in subdirectories:
        print(subdirectory)
        file_path = os.path.join(root, subdirectory)
        for v in os.listdir(file_path):
            if v.endswith('.png'):
                
                val_path=file_path + '/' + m   #create first of dic values, i.e the path
                print(img_path)
                #print(img_path)
                #imagename=ntpath.basename(imagepath)#take the name of the file from the path and save it
                val = imread(img_path)#[:,:,:IMG_CHANNELS]
                val = resize(val, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
                #X_train[n1] = img  #Fill empty X_train with values from img
                Z.append(mask)
        #train_labels.append(label)
#Convert list to array for machine learning processing        
Z = np.array(Z)
#from sklearn.model_selection import train_test_split
#x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

x_train = X
y_train = Y
# preprocess input
x_train = preprocess_input(x_train)
x_val = preprocess_input(x_val)

# define model
model = sm.Unet(BACKBONE, encoder_weights='imagenet')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mse'])

print(model.summary())


history=model.fit(x_train, 
          y_train,
          batch_size=8, 
          epochs=10,
          verbose=1,
          validation_data=(x_val))



#accuracy = model.evaluate(x_val, y_val)
#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
#plt.savefig('Training_loss.pdf')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
#plt.show()
plt.savefig('Training_Validation_loss.pdf')

#model.save('membrane.h5')


from tensorflow import keras
model = keras.models.load_model('membrane.h5', compile=False)
#Test on a different image
#READ EXTERNAL IMAGE...
test_img = cv2.imread('/home/inf-54-2020/experimental_cop/test.png', cv2.IMREAD_COLOR)       
test_img = cv2.resize(test_img, (IMG_HEIGHT, IMG_WIDTH))
test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
test_img = np.expand_dims(test_img, axis=0)

prediction = model.predict(test_img)

#View and Save segmented image
prediction_image = prediction.reshape(mask.shape)
plt.imshow(prediction_image, cmap='gray')
plt.imsave('/home/inf-54-2020/experimental_cop/saved_images/test0_segmented.jpg', prediction_image, cmap='gray')
