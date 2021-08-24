#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 14:03:19 2021

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
seed = 42
np.random.seed = seed

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

TRAIN_IMG_DIR = "/home/inf-54-2020/experimental_cop/Train_H_Final/Aug_Img/"
M_TRAIN_IMG_DIR = "/home/inf-54-2020/experimental_cop/Train_H_Final/Aug_Mask/"

#TRAIN_IMG_DIR = '/home/atte/kansio/img/'
#M_TRAIN_IMG_DIR ='/home/atte/kansio/img_mask/'

VAL_IMG_DIR = "/home/inf-54-2020/experimental_cop/Val_H_Final/Orginal_unpatched/"
M_VAL_IMG_DIR = "/home/inf-54-2020/experimental_cop/Val_H_Final/Orginal_unpatched/masks/"


train_ids = next(os.walk(TRAIN_IMG_DIR))[1] #returns all sub dirs found within this dir 
m_train_ids = next(os.walk(M_TRAIN_IMG_DIR))[1] #returns all sub dirs found within this dir 

#test_ids = next(os.walk(VAL_IMG_DIR))[1]
no_of_files = len(train_ids)
no_of_masks = len(m_train_ids)
print(no_of_files)
print(no_of_masks)
X_train = np.zeros((no_of_files, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((no_of_files, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

X_train=[]
Y_train=[]

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
np.save('/home/inf-54-2020/experimental_cop/scripts/X_train_size128.npy', X_train)

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

np.save('/home/inf-54-2020/experimental_cop/scripts/Y_train_size128.npy', Y_train)

print('masks saved into array!')
    
#convert X and Y train into numpy arrays
X_train=np.array(X_train)
print('X_train:')
print(X_train.shape)
print(X_train.size)
Y_train=np.array(Y_train)
print('Y_train:')
print(X_train.shape)
print(X_train.size)


# test images
#X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
X_test=[]
sizes_test = []
n3 = 0
for root, subdirectories, files in tqdm(os.walk(VAL_IMG_DIR)): #tqdm shows the progress bar of the for loop
    #print(root)
    for subdirectory in subdirectories:
    #    print(subdirectory)
        file_path = os.path.join(root, subdirectory)
     #   print(file_path)
        for f in os.listdir(file_path):
            if not f.endswith('.tif'):
                continue
            img_path=file_path + '/' + f   #create first of dic values, i.e the path
            #print(img_path)
            #imagename=ntpath.basename(imagepath)#take the name of the file from the path and save it
            img = imread(img_path)[:,:,:IMG_CHANNELS]
            sizes_test.append([img.shape[0], img.shape[1]])
            img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
            X_test.append(img)
            #print(' loop of X_test done!')
X_test = np.array(X_test)
np.save('/home/inf-54-2020/experimental_cop/scripts/X_test_size128.npy', X_test)

print('Test files saved into array!')
