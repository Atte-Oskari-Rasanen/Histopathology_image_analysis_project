#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 13:01:34 2021

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

cp_save_path = "/home/inf-54-2020/experimental_cop/scripts/New_model_bs128.h5"
model_segm = keras.models.load_model(cp_save_path)

im_path = "/home/inf-54-2020/experimental_cop/Train_H_Final/Train/"

path_to_img = '/home/atte/Documents/googletest.jpeg'
save_path = "/home/inf-54-2020/experimental_cop/output_segm/"
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

directory = '/home/inf-54-2020/experimental_cop/Original_Images/'
path, dirs, files = next(os.walk(directory))
file_count = len(files)
n = 0
for imagefile in os.listdir(directory):  #to go through files in the specific directory
    #print(os.listdir(directory))
    imagepath=directory + "/" + imagefile
    # if not imagefile.endswith('.tif') or imagefile.endswith('.jpg'): #exclude files not ending in .tif
    #     continue
    #print(imagepath)
    imagename=ntpath.basename(imagepath)#take the name of the file from the path and save it

    #print(imagename)
    #get the threshold
    img = Image.open(imagepath) #create a gray image of the original one using rgb2gray
    print(imagefile)
    img = np.asarray(img)
    print(img.shape)
    try:
        img_h, img_w, _ = img.shape
    #img = np.expand_dims(img, 0)
    except ValueError:
        pass
    #img = np.resize(img, (500,500))
    split_width = 128
    split_height = 128

    X_points = start_points(img_w, split_width, 0.1)
    Y_points = start_points(img_h, split_height, 0.1)
    #print(Y_points.shape)
    splitted_images = []
    
    for i in Y_points:
        for j in X_points:
            split = img[i:i+split_height, j:j+split_width]
            split = np.expand_dims(split, 0)
            #print(split.shape)
            #split = split.astype(np.uint8)
            #segm = model_segm.predict(split)
            #im = Image.fromarray(segm)
            #im.save(im_path + str(i) + str(j) +'_10x_1_remade.png')
            splitted_images.append(split) #now you have created a mask for the patch
    segm_patches = []
    i = 0
    
    for patch in splitted_images:
        #print(patch.shape)
        segm = (model_segm.predict(patch)[0,:,:,0] > 0.5).astype(np.uint8)
        #print(segm)
        #segm_ready = segm.astype(np.uint8)
        segm = model_segm.predict(patch)
        #th = threshold_otsu(segm)
        #segm_ready = (segm > th).astype(np.uint8)
        #segm=np.asarray(segm)
        #print(segm.shape)
        im = np.squeeze(segm)  #need to get rid of the channel dim, otherwise PIL gives an error
        
        #segm = np.expand_dims(segm,0)
        im = (im * 255).astype(np.uint8)
        segm_ready = (segm * 255).astype(np.uint8)
    
        #print(segm)
        im = Image.fromarray(im)
        
        #im = im.convert("L")
        #im.save(save_path + str(i) + 'patch_20x_1_remade.png')
        i += 1
        #print(type(segm))
        segm_patches.append(segm_ready)
    
    #print(segm_patches)
    #rebuild phase
    final_image = np.zeros_like(img)
    
    index = 0
    for i in Y_points:
        for j in X_points:
            final_image[i:i+split_height, j:j+split_width] = segm_patches[index]
            index += 1
    n+=1
    left = file_count - n
    #final_image = np.squeeze(final_image)  #need to get rid of the channel dim, otherwise PIL gives an error
    print(final_image.shape)
    #final_image = np.array(final_image)
    #print(final_image.shape)
    #final_image = np.expand_dims(final_image,0)
    
    #segm=np.asarray(segm)
    #im = np.squeeze(segm)  #need to get rid of the channel dim, otherwise PIL gives an error
    #im = Image.fromarray((final_image * 255).astype(np.uint8))
    im = Image.fromarray(final_image)
    im.save(save_path + imagename + '_S.png')
    print(str(n) + "th done!" + str(left) + "left...")
#scipy.misc.imsave(save_path + '20x_1_remade.png', final_image)

#im = im.convert("L")
#imsave(save_path + '20xremade.tif', final_image)
# import imagej
# ./ImageJ-linux32 --headless --console -macro ./Contrast_TH.ijm 'input directory=/home/inf-54-2020/experimental_cop/Segm_images/ output directory=/home/inf-54-2020/experimental_cop/Segm_images/TH_Otsu/ .png


print('Done!')