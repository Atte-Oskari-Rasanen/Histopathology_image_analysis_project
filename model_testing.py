#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 15:14:31 2021

@author: atte
"""

import os
import random
import numpy as np
from tqdm import tqdm 
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt

import cv2
from keras.utils import normalize
import tensorflow as tf
seed = 42
np.random.seed = seed

# IMG_WIDTH = 128
# IMG_HEIGHT = 128
# IMG_CHANNELS = 3

# VAL_IMG_DIR = "/home/inf-54-2020/experimental_cop/Val_H_Final/Images/"
# X_test=[]
# sizes_test = []
# n3 = 0
# print('starting...')
# for root, subdirectories, files in tqdm(os.walk(VAL_IMG_DIR)): #tqdm shows the progress bar of the for loop
#     #print(root)
#     for subdirectory in subdirectories:
#     #    print(subdirectory)
#         file_path = os.path.join(root, subdirectory)
#       #   print(file_path)
#         for f in os.listdir(file_path):
#             if not f.endswith('.png'):
#                 continue
#             img_path=file_path + '/' + f   #create first of dic values, i.e the path
#             #print(img_path)
#             #imagename=ntpath.basename(imagepath)#take the name of the file from the path and save it
#             img = imread(img_path)[:,:,:IMG_CHANNELS]
#             sizes_test.append([img.shape[0], img.shape[1]])
#             img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
#             X_test.append(img)
#             #print(' loop of X_test done!')
# X_test = np.array(X_test)
# print(X_test.shape)

# np.save('/home/inf-54-2020/experimental_cop/scripts/X_test_size128.npy', X_test)
X_test = np.load('/home/inf-54-2020/experimental_cop/scripts/X_test_size128.npy')
X_test = np.expand_dims(X_test, axis=0)
#upload the numpy files as well as the saved model location
#X_train = np.load('/home/inf-54-2020/experimental_cop/scripts/X_train_size128_Unet.npy')
#Y_train = np.load('/home/inf-54-2020/experimental_cop/scripts/Y_train_size128_Unet.npy')
Y_train = np.load('/home/inf-54-2020/experimental_cop/scripts/kd_Y_train_size128.npy')
X_train = np.load('/home/inf-54-2020/experimental_cop/scripts/kd_X_train_size128.npy')
#X_test = np.load('/home/inf-54-2020/experimental_cop/scripts/X_test_size128_Unet.npy')
cp_save_path = "/home/inf-54-2020/experimental_cop/scripts/kaggle_model.h5"
from tensorflow import keras

model_segm = keras.models.load_model(cp_save_path)
#im_path = "/home/inf-54-2020/experimental_cop/Earlier_YZ004_NR_G2_#15_hCOL1A1_10x_H_Final_921.png"
im_path = "/home/inf-54-2020/experimental_cop/test_precipitate.png"
#im_path = "/home/inf-54-2020/experimental_cop/test2.png" 
#im_path = '/home/inf-54-2020/experimental_cop/Train_H_Final/Aug_Img/augmented_image_253/augmented_image_253_186.png'
#im_path = '/home/inf-54-2020/experimental_cop/Train_H_Final/Aug_Img/augmented_image_253/augmented_image_253_1812.png'
#im_path = '/home/inf-54-2020/experimental_cop/Train_H_Final/Aug_Img/augmented_image_253/augmented_image_253_80.png'
save_path = "/home/inf-54-2020/experimental_cop/saved_images/reconstruction_Testdata1.png"
from matplotlib import image
from PIL import Image

img = imread(im_path)[:,:,:3]
print(img.shape)

im = resize(img, (128, 128), mode='constant', preserve_range=True)
#print(im.shape)
# asarray() class is used to convert
# PIL images into NumPy arrays
im = np.asarray(im)
im = im.astype('uint8')
im = np.expand_dims(im, axis=0)
#test_im = resize(test_im (128,128,3))
results = model_segm.predict(im)  #needs 4 dims....
print(results.shape)
results = np.squeeze(results)
#results = results.convert("L")

print(results.shape)
#results = results.convert('RGB')

#results = np.expand_dims(results, axis=0)
print(results)
print(results.size)
#from skimage.filters import threshold_otsu
#thresh = threshold_otsu(results)
#print(thresh)
#binary = results > thresh

#binary = Image.fromarray(binary)  #needs 3 dims!!!
#import matplotlib.cm as cm
#import numpy as np
#plt.imsave(im_path, binary, cmap=cm.gray) #another way of saving

#im.save(im_path)
#cv2.imwrite(save_path, binary)

cv2.imwrite(save_path, results * 255)

# preds_test_t = (results > 0.5).astype(np.uint8)

# #model = pickle.load(open(cp_save_path,"rb"))
# save_path = "/home/inf-54-2020/experimental_cop/saved_images/reconstruction_Testdata.jpg"
# print(X_test[0])
# #model = keras.models.load_model(cp_save_path)
# results = model_segm.predict(X_test[0], verbose=1)
# preds_test_t = (results > 0.5).astype(np.uint8)
# im = tf.squeeze(preds_test_t[0])
# plt.imsave(save_path, im)
print('All done!')
#fig = plt.figure()
# plt.ioff()
# plt.plot(preds_test_t[0])
# plt.savefig(save_path)
# plt.close(fig)

# ix = random.randint(0, len(preds_test_t))

#imshow(X_test[0])
#plt.savefig(save_path)

# print(preds_test_t)
# arr = preds_test_t[0]
# im = Image.fromarray((arr * 255).astype(np.uint8))
# cv2.imwrite(save_path, im)
