#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 20:07:06 2021

@author: atte
"""

import os
import random
import numpy as np
from tqdm import tqdm 
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
seed = 42
np.random.seed = seed
import PIL
from PIL import Image, ImageOps
import cv2
from keras.utils import normalize

import tensorflow as tf
import os
import random
import numpy as np
from tensorflow import keras
from tifffile import imsave
import ntpath

#####
#apply this to large images, train the model with these smaller patches
#when predicting on large images, break the image into smaller patches like this,
#then apply the processes like model.predict on these arrays, append into segm_images
#and then save as a whole slide image
import cv2
import scipy.misc

cp_save_path = "/home/inf-54-2020/experimental_cop/scripts/Model_512.h5"
model_segm = keras.models.load_model(cp_save_path)

im_path = "/home/inf-54-2020/experimental_cop/batch3/"

path_to_img = '/home/atte/Documents/googletest.jpeg'
save_path = "/home/inf-54-2020/experimental_cop/All_imgs_segm/"
#img = cv2.imread(im_path + '20x_1_H_Final_1.jpg')
#img = cv2.imread(save_path + 'YZ004_NR_G2_#15_hCOL1A1_10x__1_H_Final.jpg')

#If tif:
#img = Image.open('/home/inf-54-2020/experimental_cop/Hu_D_45_min_10X.tif')
#img = cv2.imread(save_path + 'test2.png')


#t = cv2.imread('/home/inf-54-2020/experimental_cop/Train_H_Final/Aug_Img/augmented_image_14/augmented_image_14_184.png')
#t = np.resize(t,(128,128,3))
#t = np.expand_dims(t,0)
# r = model_segm(t)
# r.save(save_path + 'r.png')


def start_points(size, split_size, overlap=0):
    points = [0]
    stride = int(split_size * (1-overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            points.append(size - split_size)
            break
        else:
            points.append(pt)
        counter += 1
    return points
split_width = 512
split_height = 512

def gen_patches(p, f, n_path):
    img = cv2.imread(p + f)
    # try:
    img_h, img_w, _ = img.shape
    print(img.shape)
    # #img = np.expand_dims(img, 0)
    # except ValueError:
    #     pass
    X_points = start_points(img_w, split_width, 0.1)
    Y_points = start_points(img_h, split_height, 0.1)

    #print(patches_img.shape)
    a = 0
    b = 0
    for i in Y_points:
        a += 1
        for j in X_points:
            single_patch_img = img[i:i+split_height, j:j+split_width]
            #single_patch_img = np.expand_dims(single_patch_img, 0)
            #print(single_patch_img.shape)
            #print(split.shape)
            #cv2.imwrite(new_path + f_name + '_' + str(i ) +str(j) +'.png', img)
            if not cv2.imwrite(n_path + f_name + '_'+ str(a)+str(b)+'.png', single_patch_img):
                raise Exception("Could not write the image")
            b += 1

path = "/home/inf-54-2020/experimental_cop/Train_H_Final/Aug_Img/"

for i, f in enumerate(sorted(os.listdir(path))):
    print(f)
    #if not os.path.isdir(f): #check if a dir with the folder name exists
    f_name=f.rsplit('.', 1)[0]
    n_path = path + f_name + '/'  #new path into which the image is saved into
    if os.path.isdir(n_path): #if the directory already exists, then skip!
        continue

    os.mkdir(n_path)  #create a new dir with the file name 
    print('made the directory!')
    gen_patches(path, f, n_path)  #call for the function that generates patches and saves them into the new dir
print('Done!')


