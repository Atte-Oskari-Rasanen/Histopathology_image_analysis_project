#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 17:14:25 2021

@author: atte
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from skimage.filters import threshold_otsu
from PIL import Image, ImageFilter
from skimage import measure, filters
import scandir
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage as ndi
import os
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

import sys


def colocalise(hunu_im, col1a1_im):
    hunu_im = cv2.imread(hunu_im,0)
    print(hunu_im.shape)
    col1a1_im = cv2.imread(col1a1_im,0)
    ret2,hunu_im = cv2.threshold(hunu_im,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret2,col1a1_im = cv2.threshold(col1a1_im,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    print("COL1A1 SHAPE: " + str(col1a1_im.shape))
    # hunu_im = cv2.bitwise_not(hunu_im)
    h,w = col1a1_im.shape
    hunu_im = Image.fromarray(hunu_im)
    hunu_im = hunu_im.resize((w,h))
    hunu_im = np.asarray(hunu_im)
    print("HUNU SHAPE: " + str(hunu_im.shape))
    cnts, _ = cv2.findContours(col1a1_im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #get the contours of the col1a1 
    
    out_mask = np.zeros_like(hunu_im)
    
    
    #use this when applying mask to the image of nuclei
    cv2.drawContours(out_mask, cnts, -1, 255, cv2.FILLED, 1)                                        


    out=hunu_im.copy()
    out[out_mask == 0] = 255 #makes nuclei white on the black background
    out = cv2.bitwise_not(out)
    return(out)
    print('colocalised image created!')

main_dir = sys.argv[1]
coloc_dir = main_dir + '/Coloc'

try:
    os.mkdir(coloc_dir)
except OSError:
    print ("Failed to create directory %s " % coloc_dir)
else:
    print ("Succeeded at creating the directory %s " % coloc_dir)

ids= []
segm_TH_dirs = []
all_ims_paths = []
for (dirpath, dirnames, filenames) in os.walk(main_dir):
    all_ims_paths += [os.path.join(dirpath, file) for file in filenames]

file_pairs = {} #key: hunu_ws_th file, value: col1a1_th
print('STARTING...')
for f in all_ims_paths:
    # print(f)
    filename = os.path.basename(f)
    if 'WS' in filename:
        # my_search_string = os.path.basename(f)
        for f2 in all_ims_paths:
            filename2 = os.path.basename(f2)
            print('filename2: '+filename2)
            if 'col1a1' in filename2 and 'TH' in filename2 and filename[:18] in filename2:
            # if my_search_string[:18] in os.path.basename(f2):
                file_pairs[f] = f2


#Now we perform colocalisation:

for hunu, col1a1 in file_pairs.items():
    print('hunu: ' +hunu)
    filename = os.path.basename(hunu)
    filename = filename.split('.')[0]
    print('col1a1: ' +col1a1)
    coloc_im = colocalise(hunu,col1a1)
    # coloc_im = cv2.bitwise_not(coloc_im)

    cv2.imwrite(coloc_dir +'/' + filename + "_Coloc.png",coloc_im)
    print('SAVED :' + coloc_dir +'/' + filename + "_Coloc.png" )
print(coloc_dir)
print("ALL COLOCALISED")
