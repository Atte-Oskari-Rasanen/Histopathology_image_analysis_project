#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 18:23:45 2021

@author: atte
"""

import numpy as np
import cv2
import os
from PIL import Image
#when you transform kaggle (rgba format) images into numpy arrays (first import with normal
#dims so 4, then transform into rgb via cv2), and when you transform them back and 
#put them into the right dir format, keep in mind that masks dont have the inside contours of nuclei
#filled. This script fills in the contours and replaces the old masks with the new ones

mask_dir = '/home/inf-54-2020/experimental_cop/Train_H_Final/Masks_by_batches/org_kaggle/'
mask_dir = '/home/inf-54-2020/experimental_cop/Val_H_Final/Good_full_masks/Masks/data_kaggle/'
# mask_dir = '/home/atte/Desktop/org_kaggle/'
# out = '/home/atte/Desktop/org_kaggle/'
from skimage import img_as_uint
from skimage import io

for imagefile in os.listdir(mask_dir):  #to go through files in the specific directory
    #print(os.listdir(directory))
    imagepath=mask_dir + imagefile
    #print(imagepath)

    image = cv2.imread(imagepath)
    #print(image.shape)
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        
        for c in cnts:
            cv2.drawContours(gray,[c], 0, (255,255,255), -1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,20))
        opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=2)
        #g = (gray * 255).astype(np.uint8)
        io.imsave(imagepath, img_as_uint(gray))
    except cv2.error:
        pass
