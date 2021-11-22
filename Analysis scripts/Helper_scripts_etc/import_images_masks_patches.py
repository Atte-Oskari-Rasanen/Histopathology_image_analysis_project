import tensorflow as tf
import os
import random
import numpy as np
from tqdm import tqdm 
import matplotlib
matplotlib.use('Agg')

from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import sys
import cv2
from PIL import Image
import glob

import numpy as np


# print(a.shape)
#all images need to be imported, then cut into patches of the specific size. These patches
#are saved into the numpy arrays of X_train (original images) and Y_train (masks). 

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

def gen_patches(im, split_height, split_width):
    img = im
    img_h, img_w, _ = img.shape
    X_points = start_points(img_w, split_width, 0.1)
    Y_points = start_points(img_h, split_height, 0.1)
    
    #create an empty numpy array into which you can put all the patches of the single image
    #and return this to the import function
    patches_list = []
    #print(patches_img.shape)
    a = 0
    b = 0
    for i in Y_points:
        a += 1
        for j in X_points:
            single_patch_img = img[i:i+split_height, j:j+split_width]
            print(single_patch_img.shape)
            # single_patch_img = resize(single_patch_img, (single_patch_img, split_height,split_width ), mode='constant', preserve_range=True)
            patches_list.append(single_patch_img)
            print(single_patch_img.shape)
    patches_array = np.array(patches_list)
    return(patches_array)

import functools
def combine_dims(a, i=0, n=1): #combines dimension n and n+1
    """
    Combines dimensions of numpy array `a`, 
    starting at index `i`,
    and combining `n` dimensions
    """
    s = list(a.shape)
    combined = functools.reduce(lambda x,y: x*y, s[i:i+n+1])
    final_shape = np.reshape(a, s[:i] + [combined] + s[i+n+1:])
    return final_shape



def import_images(TRAIN_PATH, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    split_height = int(IMG_HEIGHT)
    split_width = int(IMG_WIDTH)
    # train_ids = next(os.walk(TRAIN_PATH))
    # X_train = np.zeros((len(train_ids), None, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    list_patched_ims = []
    n1 = 0
    print('starting')
    image_names = glob.glob(TRAIN_PATH + '*.png')
    image_names.sort()
    for im in image_names:
        img = cv2.imread(im)[:,:,:IMG_CHANNELS]
        #if image size is smaller than the designed size of the patches, simply resize to the 
        #patch size, save to the array and continue onto the next image
        img_h, img_w, _ = img.shape
        img_h = int(img_h)
        img_w = int(img_w)
        if img_h <= IMG_HEIGHT or img_w <= IMG_WIDTH:
            img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
            #need to expand dimensions by 1 so that the dimensions match with the other arrays
            #prior to concatenating them
            img = np.expand_dims(img, axis = 0)
            list_patched_ims.append(img)
            # print(str(n1) + ' loop of X_train done!')
            continue
        # img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        img_patch = gen_patches(img, split_width, split_height)
        list_patched_ims.append(img_patch)

        n1 += 1
        
    X_train = np.concatenate(list_patched_ims, axis=0)
    
    # X_train = np.array(X_train, dtype=np.float32)
    #At the moment the numpy array shape is: (no of images, no of patches per image,
    #img height, img width, channels). Need to combine the first two dims:
    # X_train = combine_dims(X_train, 0) # combines dimension 0 and 1
    print(X_train.shape)
    return(X_train)


    print('Images saved into array!')

from skimage.filters import threshold_otsu

def import_masks(MASK_PATH, IMG_HEIGHT, IMG_WIDTH):
    # train_ids = next(os.walk(MASK_PATH))
    # Y_train = np.zeros((len(train_ids), None, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    list_patched_masks = []
    n2 = 0
    mask_names = glob.glob(MASK_PATH + '*.png')
    mask_names.sort()
    for mask in mask_names:
        mask = cv2.imread(mask,0)
        thresh = threshold_otsu(mask)
        (T, thresh_im) = cv2.threshold(mask, thresh, 255,
        	cv2.THRESH_BINARY)
        mask = np.expand_dims(thresh_im, axis = 2)
        # print(mask.shape)
        mask_h, mask_w,_ = mask.shape
        if mask_h <= IMG_HEIGHT or mask_w <= IMG_WIDTH:
            mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
            # Y_train.astype('float32')
            mask = np.expand_dims(mask, axis = 0)
            list_patched_masks.append(mask)
            # Y_train.append(mask)
            print(str(n2) + ' loop of Y_train done!')
            continue
        mask_patches = gen_patches(mask, IMG_HEIGHT, IMG_WIDTH)
        list_patched_masks.append(mask_patches)

        print(str(n2) + ' loop of Y_train done!')
        n2 += 1
    # Y_train = np.array(Y_train, dtype=np.float32)
    Y_train = np.concatenate(list_patched_masks, axis=0)
    # Y_train = combine_dims(Y_train, 0) # combines dimension 0 and 1
    print(Y_train.shape)
    return(Y_train)


    Y_train=np.array(Y_train)
    return(Y_train)
    # Y_train = np.expand_dims(Y_train, axis =2)
    print(Y_train.shape)
    print(len(Y_train))
    print('Masks saved into array!')

def import_kaggledata(kaggle_data_path, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    train_ids = next(os.walk(kaggle_data_path))[1]

    X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    # X_train = []
    # Y_train  = []
    print('Resizing training images and masks')
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):   
        path = kaggle_data_path + id_
        img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]  
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_train[n] = img  #Fill empty X_train with values from img
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = imread(path + '/masks/' + mask_file)
            mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',  
                                          preserve_range=True), axis=-1)
            mask = np.maximum(mask, mask_)  
            
    Y_train[n] = mask   
    return X_train, Y_train
