#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 16:11:38 2021

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

import tensorflow as tf
import os
import random
import numpy as np
import keras
#from tifffile import imsave
import ntpath

#####
#apply this to large images, train the model with these smaller patches
#when predicting on large images, break the image into smaller patches like this,
#then apply the processes like model.predict on these arrays, append into segm_images
#and then save as a whole slide image

cp_save_path = "/cephyr/NOBACKUP/groups/snic2021-23-496/scripts/working_models/kaggle_model_size128.h5"
model_segm = keras.models.load_model(cp_save_path)

# im_path = "/home/inf-54-2020/experimental_cop/batch3/"
im_path = '/cephyr/NOBACKUP/groups/snic2021-23-496/batch3/Hu_D_30_min_10X.tif'
# path_to_img = '/home/atte/Documents/googletest.jpeg'
# save_path = "/home/inf-54-2020/experimental_cop/All_imgs_segm/"
save_path = '/cephyr/NOBACKUP/groups/snic2021-23-496/All_imgs_segm/'
print(save_path)

def Segment_img(img_path,model_segm):
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
    imagename = img_path.split('/')[-1]
    imagename = imagename.split('.')[0]
    print(imagename)
    img = Image.open(img_path)
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
    splitted_images = []

    for i in Y_points:
        for j in X_points:
            split = img[i:i+split_height, j:j+split_width]
            split = np.expand_dims(split, 0)
            print(split.shape)
            splitted_images.append(split) #now you have created a mask for the patch
    segm_patches = []
    i = 0
    
    for patch in splitted_images:
        print(patch.shape)
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
    #final_image = np.squeeze(final_image)  #need to get rid of the channel dim, otherwise PIL gives an error
    print(final_image.shape)
    #final_image = np.array(final_image)
    #print(final_image.shape)
    #final_image = np.expand_dims(final_image,0)
    
    #segm=np.asarray(segm)
    #im = np.squeeze(segm)  #need to get rid of the channel dim, otherwise PIL gives an error
    #im = Image.fromarray((final_image * 255).astype(np.uint8))
    #im = Image.fromarray(final_image)
    print(type(im))
    return(im)
    #cv2.imwrite("/cephyr/NOBACKUP/groups/snic2021-23-496/All_imgs_segm/S_Hunu_30min.png", final_image)
    
#    plt.imsave(imagename + '_512S2.png', final_image)

    #im.save(save_path + imagename + '_512S2.png')

#im = im.convert("L")
#imsave(save_path + '20xremade.tif', final_image)
# import imagej
# ./ImageJ-linux32 --headless --console -macro ./Contrast_TH.ijm 'input directory=/home/inf-54-2020/experimental_cop/Segm_images/ output directory=/home/inf-54-2020/experimental_cop/Segm_images/TH_Otsu/ .png
img = Segment_img(im_path, model_segm)
img.save(save_path + 'S_Hunu_30min_128.png')
print('Done!')
