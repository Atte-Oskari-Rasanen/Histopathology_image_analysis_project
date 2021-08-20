#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 23:21:19 2021

@author: atte
"""

from patchify import patchify
import cv2
import numpy as np
#import pathlib
import os

from PIL import Image

def gen_patches(path, file, new_path):
    
    img = cv2.imread(path + file)
    patches_img = patchify(img, (244,244,3), step=244)
    print(patches_img.shape)
    #print(patches_img.shape)
    for i in range(patches_img.shape[0]): #go through the height
        for j in range(patches_img.shape[1]): #go through the width
            single_patch_img = patches_img[i,j,:,:]
            print('file type:')
            #single_patch_img = patches_img[i, j, 0, :, :, :]
            print(single_patch_img.shape)
            single_patch_img = single_patch_img.mean(axis=0) #remove first dimension
            print('edited:')
            print(single_patch_img.shape)
            
            f_name=file.rsplit('.', 1)[0]
            #print(single_patch_img)
            print(new_path + f_name + '_'+ str(i)+str(j)+'.png')
            #you need to convert the image into uint8 type. No floats allowed when saving the image
            img = single_patch_img.astype(np.uint8)
            #cv2.imwrite(new_path + f_name + '_' + str(i ) +str(j) +'.png', img)
            if not cv2.imwrite(new_path + f_name + '_'+ str(i)+str(j)+'.png', single_patch_img):
                raise Exception("Could not write the image")
#path_list = ["/home/inf-54-2020/experimental_cop/H_final/Images/", "/home/inf-54-2020/experimental_cop/H_final/Masks/",
#              "/home/inf-54-2020/experimental_cop/Val_H_Final/Images/"]

#path = "/home/inf-54-2020/experimental_cop/H_final/Images/"
path = '/home/inf-54-2020/experimental_cop/Train_H_Final/Aug_Img/'
#for p in path_list: #iterate over the paths in the path list
    #print(p)
#for f in os.listdir(path):  #iterate over files in the path at hand
for i, f in enumerate(sorted(os.listdir(path))):
    print(f)
    #if not os.path.isdir(f): #check if a dir with the folder name exists
    f_name=f.rsplit('.', 1)[0]
    n_path = path + f_name + '/'  #new path into which the image is saved into
    os.makedirs(n_path, exist_ok=True)  #create a new dir with the file name 
    print('made the directory!')
    gen_patches(path, f, n_path)  #call for the function that generates patches and saves them into the new dir
