#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 14:57:56 2021

@author: atte
"""

import tensorflow as tf
import os
import random
import numpy as np
 
from tqdm import tqdm 
import sys
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib
matplotlib.use('Agg')
import cv2
import matplotlib.pyplot as plt

IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 3


TRAIN_IMG_DIR = sys.argv[2]
M_TRAIN_IMG_DIR = sys.argv[3]

img_dir_id = [] #list of dir ids containing patches of the certain image
ind_im_ids = [] #create an empty list for the ids of the individual images found in the subdir
n1 = 0

X_train = []
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
np.save('/cephyr/NOBACKUP/groups/snic2021-23-496/X_train_kagl_own_s512.npy', Y_train)

#np.save('/home/inf-54-2020/experimental_cop/scripts/Y_train_size128.npy', Y_train)
print(Y_train.shape)
print(Y_train)
print('masks saved into array!')
