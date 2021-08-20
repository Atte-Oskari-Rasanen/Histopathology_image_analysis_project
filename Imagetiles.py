#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 12:00:02 2021

@author: atte
"""
import cv2
import numpy as np
#import pathlib
import os
from skimage.transform import resize
import math

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import tensorflow as tf
#from keras.utils import normalize
from matplotlib import pyplot as plt
from patchify import patchify

image= Image.open('/home/atte/Documents/images_qupath2/H_final/YZ004 NR G2 #15 hCOL1A1 10x _1_H_Final.jpg')
img = np.asarray(image)

patch_size = 567
img = np.resize(img, (1024,768,3))

for i in range(img.shape[0]):
    patches = patchify = patchify(img, (256, 256), step=)

    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):

for i in range(0, img.shape[0], 567):
    for j in range(0, img.shape[1], 567):
        single_patch = img[i:i+patch_size, j:j+patch_size] #extract patch size 
        print(np.size(single_patch))
        print(single_patch.shape)
        patch = Image.fromarray(single_patch, 'RGB')
        patch.save("filename.png")


path = '/home/atte/'
M = img.shape[0]//6
N = img.shape[1]//6

tiles = [img[x:x+M,y:y+N] for x in range(0,img.shape[0],M) for y in range(0,img.shape[1],N)]
#img = Image.fromarray(tiles[0])
for i in tiles:
    img = Image.fromarray(tiles[i])
    name = path + str(i) + '.png'
    img.save(name)
    
img.show()

def crop(path, input, height, width, k, page, area):
    im = Image.open(input)
    imgwidth, imgheight = im.size
    for i in range(0,imgheight,height):
        for j in range(0,imgwidth,width):
            box = (j, i, j+width, i+height)
            a = im.crop(box)
            try:
                o = a.crop(area)
                o.save(os.path.join(path,"PNG","%s" % page,"IMG-%s.png" % k))
            except:
                pass
            k +=1

n = range(1,20)
lista = [n]
valids = []

#take dimensions from both images, check which are divisible, then save both lists, find the 
#lowest common number with which they are divisible by and determine the patch dims based on this

h, w = image.shape[:2]

def patch_size(img):
    h, w = img.shape[:2]

    height_numbers = [x for x in list(range(1,10)) if h % x == 0]
    width_numbers = [x for x in list(range(1,10)) if w % x == 0]
    
    #get the smallest possible dimensions for the patches so iterate from end to beginning
    new_dims = []
    for n in reversed(height_numbers):
        if n in width_numbers:
            print(n)
            #get new image dims:
            patch_h = int(h/n)
            new_dims.append(path_h)
            patch_w = int(w/n)
            new_dims.append(patch_w)
            break
    #img = np.resize(img, (new_h, new_w))
    patches_img = patchify(img, (new_h,new_w,3), step=new_w)

    return(new_dims)
    
img = patch_size()
def pad_images_to_same_size(img):
    """
    :param images: sequence of images
    :return: list of images padded so that all images have same width and height (max width and height are used)
    """
    width_max = 0
    height_max = 0
    h, w = img.shape[:2]
    width_max = max(width_max, w)
    height_max = max(height_max, h)

    h, w = img.shape[:2]
    diff_vert = height_max - h
    pad_top = diff_vert//2
    pad_bottom = diff_vert - pad_top
    diff_hori = width_max - w
    pad_left = diff_hori//2
    pad_right = diff_hori - pad_left
    img_padded = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
    assert img_padded.shape[:2] == (height_max, width_max)
    images_padded.append(img_padded)

    return images_padded
img = pad_images_to_same_size(image)

constant= cv2.copyMakeBorder(image.copy(),10,10,10,10,cv2.BORDER_CONSTANT)
img = Image.fromarray(constant)
img.show()



constant.imshow()
def apply_padding(image3, tile_size, padding="CONSTANT"):
    imagesize = tf.shape(image3)[0:2]
    padding_ = (tf.ceil(imagesize / tile_size)) * tile_size - imagesize
    return tf.pad(image3, [[0, padding_[0]], [0, padding_[1]], [0, 0]], padding)

image = pad_image_to_tile_multiple(image, [525,525])


tiles = split_image(image, [28, 28])

def prediction(model, image, patch_size):
    segm_img = np.zeros(image.shape[:2])  #Array with zeros to be filled with segmented values
    patch_num=1
    for i in range(0, image.shape[0], 256):   #Steps of 256
        for j in range(0, image.shape[1], 256):  #Steps of 256
                
            #print(i, j)
            single_patch = image[i:i+patch_size, j:j+patch_size]#extract patch size 
            single_patch_norm = np.expand_dims(normalize(np.array(single_patch), axis=1),2)
            single_patch_shape = single_patch_norm.shape[:2]
            single_patch_input = np.expand_dims(single_patch_norm, 0)
            #single_patch_prediction = (model.predict(single_patch_input)[0,:,:,0] > 0.5).astype(np.uint8)
            segm_img[i:i+single_patch_shape[0], j:j+single_patch_shape[1]] += cv2.resize(single_patch_prediction, single_patch_shape[::-1])
          
            print("Finished processing patch number ", patch_num, " at position ", i,j)
            patch_num+=1
    return segm_img
print(segm_img)
img = cv2
# =============================================================================
# #for reconstructing the image if need be
# def unsplit_image(tiles4, image_shape):
#     tile_width = tf.shape(tiles4)[1]
#     serialized_tiles = tf.reshape(tiles4, [-1, image_shape[0], tile_width, image_shape[2]])
#     rowwise_tiles = tf.transpose(serialized_tiles, [1, 0, 2, 3])
#     return tf.reshape(rowwise_tiles, [image_shape[0], image_shape[1], image_shape[2]])
# image = unsplit_image(tiles, tf.shape(image))
# 
# =============================================================================
#crop a batch of images (-1, X, Y, 3) in N pieces:
crops = tf.reshape(tensor_images, (-1, N, tensor_images.shape[1]//N, N, tensor_images.shape[2]//N, tensor_images.shape[3]))
crops = tf.transpose(crops, [0, 1, 3, 2, 4, 5])

#remove padding
image = image[0:original_size[0], 0:original_size[1], :]


#Check the solution like this:
# =============================================================================
# def show_images(segs, x, y):
#   fig, axs = plt.subplots(x, y, figsize=(x*2, y*2))
#   for i in range(x):
#     for j in range(y):
#       axs[i, j].imshow(segs[i][j], cmap=plt.cm.binary, vmin=0, vmax=1)
#   plt.show()
#   plt.close()
# tensor_images = tf.convert_to_tensor(image_batch, dtype=tf.float32)
# =============================================================================
crops = tf.reshape(tensor_images, (-1, 8, tensor_images.shape[1]//8, 8,
tensor_images.shape[2]//8, tensor_images.shape[3]))
crops = tf.transpose(crops, [0, 1, 3, 2, 4, 5])
show_images(crops.numpy()[0], 8, 8)

