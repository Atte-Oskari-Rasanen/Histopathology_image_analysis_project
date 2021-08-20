#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 22:50:21 2021

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

#upload the numpy files as well as the saved model location
X_train = np.load('/home/inf-54-2020/experimental_cop/scripts/X_train_size128_Unet.npy')
Y_train = np.load('/home/inf-54-2020/experimental_cop/scripts/Y_train_size128_Unet.npy')
X_test = np.load('/home/inf-54-2020/experimental_cop/scripts/X_test_size128_Unet.npy')

cp_save_path = "/home/inf-54-2020/experimental_cop/scripts/Model_for_nuclei.pickle/New_model.h5"
from tensorflow import keras

model = keras.models.load_model(cp_save_path)

#model = pickle.load(open(cp_save_path,"rb"))

#model = keras.models.load_model(cp_save_path)

idx = random.randint(0, len(X_train))

#take the model and predict on random images
preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

 
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

saved_path1 = '/home/inf-54-2020/experimental_cop/saved_images/test.png'
saved_path2 = '/home/inf-54-2020/experimental_cop/saved_images/test_mask.png'
saved_path3 = '/home/inf-54-2020/experimental_cop/saved_images/test_pred.png'

# Perform a sanity check on some random training samples
ix = random.randint(0, len(preds_train_t))
imshow(X_train[ix])
plt.savefig(saved_path1)
#plt.show()
imshow(np.squeeze(Y_train[ix]))
plt.savefig(saved_path2)
#plt.show()
imshow(np.squeeze(preds_train_t[ix]))
plt.savefig(saved_path3)
#plt.show()
saved_path1 = '/home/inf-54-2020/experimental_cop/saved_images/test_Xtrain.png'
saved_path2 = '/home/inf-54-2020/experimental_cop/saved_images/test_Ytrain.png'
saved_path3 = '/home/inf-54-2020/experimental_cop/saved_images/test_pred2.png'


# Perform a sanity check on some random validation samples
ix = random.randint(0, len(preds_val_t))
imshow(X_train[int(X_train.shape[0]*0.9):][ix])
plt.savefig(saved_path1)

#plt.show()
imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]))
plt.savefig(saved_path2)

#plt.show()
imshow(np.squeeze(preds_val_t[ix]))
plt.savefig(saved_path3)

#plt.show()
