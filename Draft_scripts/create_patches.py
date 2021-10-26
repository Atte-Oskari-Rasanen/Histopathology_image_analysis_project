#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 18:57:49 2021

@author: atte
"""
from patchify import patchify
import cv2
import numpy as np
#import pathlib
import os
from skimage.transform import resize
import tensorflow as tf
tf.to_int=lambda x: tf.cast(x, tf.int32)
from imageio import imread

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# =============================================================================
# def pad_image_to_tile_multiple(image3, tile_size, padding="CONSTANT"):
#     imagesize = tf.shape(image3)[0:2]
#     print('imagesize:')
#     print(imagesize)
#     padding_ = float((imagesize / tile_size) * tile_size - imagesize)
#     print(padding)
#     return tf.pad(image3, [[0, padding_[0]], [0, padding_[1]], [0, 0]], padding)
# 
# 
# =============================================================================
def gen_patches(path, file, new_path):
    print (path + file)
    #try:
    img= Image.open(path + file)

    #img= Image.open(path + file)
    #except IsADirectoryError: #if the directory already exists, the code may give an error, ignore this
    #    pass
    #img = np.array(img.convert("RGB")) #convert to RBG before trying to resize, otherwise gives an error
    img = np.asarray(img) #into np array
    print(img.shape)
    img = np.resize(img, (5760,5670,3))

    #img = pad_image_to_tile_multiple(img, [525,525]) #generate padding so that the tiles can be created
    #img = Image.fromarray(img)
    patches_img = patchify(img, (576,576,3), step=576)
    for i in range(patches_img.shape[0]): #go through the height
        for j in range(patches_img.shape[1]): #go through the width
            single_patch_img = patches_img[i,j,:,:]
            #print('file type:')
            #single_patch_img = patches_img[i, j, 0, :, :, :]
            #print(single_patch_img.shape)
            #single_patch_img = single_patch_img.mean(axis=0) #remove first dimension
            #print('edited:')
            #print(single_patch_img.shape)
            
            f_name=file.rsplit('.', 1)[0]
            #print(single_patch_img)
            #print(new_path + f_name + '_'+ str(i)+str(j)+'.png')
            #you need to convert the image into uint8 type. No floats allowed when saving the image
            img = single_patch_img.astype(np.uint8)
            cv2.imwrite(new_path + f_name + '_' + str(i) +str(j) +'.png', img)
            print('saved to:' + new_path)
            #if not cv2.imwrite(new_path + f_name + '_'+ str(i)+str(j)+'.png', img):
             #   raise Exception("Could not write the image")
#path_list = ["/home/inf-54-2020/experimental_cop/H_final/Images/", "/home/inf-54-2020/experimental_cop/H_final/Masks/",
#              "/home/inf-54-2020/experimental_cop/Val_H_Final/Images/"]

path = "/home/inf-54-2020/experimental_cop/Train_H_Final/Aug_Mask/"
#print(path)
#for p in path_list: #iterate over the paths in the path list
    #print(p)
#for f in os.listdir(path):  #iterate over files in the path at hand
for i, f in enumerate(sorted(os.listdir(path))):
    print(f)
#if f.endswith(".tif" or ".jpg"):
 #   print(f)
    #if not os.path.isdir(f): #check if a dir with the folder name exists
    #try:
    f_name=f.rsplit('.', 1)[0]
    #except IsADirectoryError: #if the directory already exists, the code may give an error, ignore this
     #   pass
    n_path = path + f_name + '/'  #new path into which the image tile is saved into
    print('n_path:' + n_path)
    os.makedirs(n_path, exist_ok=True)  #create a new dir with the file name 

    gen_patches(path, f, n_path)  #call for the function that generates patches and saves them into the new dir


#import VGG16
#Reconstructing the image (used after segmentation)
#cv2.imwrite('patches/images/' + 'image_' + '_'+ str(i).zfill(2) + '_' + str(j).zfill(2) + '.png', single_patch_img)



# =============================================================================
# #need to create a separate directory for each image!
# img = cv2.imread('/home/inf-54-2020/experimental_cop/Val_H_Final/Images/YZ004_NR_G2_#15_hCOL1A1_20x_2_H_Final.tif')
# dims = img.shape
# dims
# h, w = img.shape[:2]
# 
# patches_img = patchify(img, (500,500,3), step=500)
# 
# #print(patches_img.shape)
# for i in range(patches_img.shape[0]):
#     for j in range(patches_img.shape[1]):
#         single_patch_img = patches_img[i, j, 0, :, :, :]
#         if not cv2.imwrite('/home/inf-54-2020/experimental_cop/test/' + 'testfile' + str(i)+str(j)+'.png', single_patch_img):
#             raise Exception("Could not write the image")
# 
# =============================================================================
