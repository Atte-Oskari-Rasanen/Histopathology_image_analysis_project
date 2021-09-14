#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 16:56:14 2021

@author: atte
"""

import albumentations as A
import cv2

transform = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import AffineTransform, warp
from skimage import io, img_as_ubyte
import random
import os
from scipy.ndimage import rotate
from tqdm import tqdm
import albumentations as A
images_to_generate=2000
import cv2
from skimage.io import imread, imshow
from skimage.transform import resize

#('/home/inf-54-2020/experimental_cop/Train_H_Final/Train/20x_1_H_Final.jpg',

# '/home/inf-54-2020/experimental_cop/Train_H_Final/Masks/10x__1_H_Final.tif')

IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 3

i_m_path='/home/inf-54-2020/experimental_cop/kaggle_data/' #path to original images
masks_path = '/home/inf-54-2020/experimental_cop/Train_H_Final/Masks/'
img_augmented_path='/home/inf-54-2020/experimental_cop/kaggle_aug_img/' # path to store aumented images
msk_augmented_path="/home/inf-54-2020/experimental_cop/kaggle_aug_mask/" # path to store aumented images
images=[] # to store paths of images from folder
masks=[]

aug = A.Compose([
    A.VerticalFlip(p=0.5),              
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=1),
    A.Transpose(p=1),
    #A.ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
    A.GridDistortion(p=1)
    ]
)


#get the images and corresponding masks, save into a list
img_ids = next(os.walk(i_m_path))[1]
for n, id_ in tqdm(enumerate(img_ids), total=len(img_ids)):   
    path = i_m_path + id_
    original_image = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]  
    original_image = resize(original_image, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    images.append(original_image)
    i=1   # variable to iterate till images_to_generate
    
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask = imread(path + '/masks/' + mask_file)
        mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    masks.append(mask)

while i<=images_to_generate: 
    number = random.randint(0, len(images)-1)  #Pick a number to select an image & masks
    image = images[number]
    filename = os.path.basename(image)
    print(filename)

    print(image)
    #if any(image in s for s in mask):

    mask = masks[number]
    print(image, mask)
    #image=random.choice(images) #Randomly select an image name
    original_image = io.imread(image)
    original_mask = io.imread(mask)
    
    augmented = aug(image=original_image, mask=original_mask)
    transformed_image = augmented['image']
    transformed_mask = augmented['mask']

        
    new_image_path= "%s/augmented_image_%s.png" %(img_augmented_path, i)
    new_mask_path = "%s/augmented_mask_%s.png" %(msk_augmented_path, i)
    io.imsave(new_image_path, transformed_image)
    io.imsave(new_mask_path, transformed_mask)
    i =i+1


# =============================================================================
# for im in os.listdir(images_path):  # read image name from folder and append its path into "images" array     
#     #print(im)
#     images.append(os.path.join(images_path,im))
# for msk in os.listdir(masks_path):  # read image name from folder and append its path into "images" array     
#     #print(msk)
#     masks.append(os.path.join(masks_path,msk))
# 
# =============================================================================


#random.seed(42)

