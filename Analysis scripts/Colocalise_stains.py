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
    print("COL1A1 SHAPE: " + str(col1a1_im.shape))
    # hunu_im = cv2.bitwise_not(hunu_im)
    h,w = col1a1_im.shape
    hunu_im = Image.fromarray(hunu_im)
    hunu_im = hunu_im.resize((w,h))
    hunu_im = np.asarray(hunu_im)
    print("HUNU SHAPE: " + str(hunu_im.shape))
    hunu_im = cv2.bitwise_not(hunu_im)
    
    # hunu_im = hunu_im.reshape(w,h)
    # cv2.imwrite('/home/atte/Documents/PD_images/batch6/col1a1.png', col1a1_im)
    cnts, _ = cv2.findContours(col1a1_im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #get the contours of the col1a1 
    #_ , contours, _ = cv2.findContours(threshInv,2,1)            
    # contours = sorted(cnts, key=cv2.contourArea)            #get the largest contour
    
    out_mask = np.zeros_like(hunu_im)
    
    #draw contours of col1a1 image onto the hunu one
    # cont = cv2.drawContours(hunu_im, cnts, -1, (0, 0, 255), 2) #-1 means draw all contours, red color, 2 is the width of contour line
    
    #use this when applying mask to the image of nuclei
    cv2.drawContours(out_mask, cnts, -1, 255, cv2.FILLED, 1)                                        

#cv2.drawContours(Img, cnts, -1, (0, 0, 255), 2) #-1 means draw all contours, red color, 2 is the width of contour line

    out=hunu_im.copy()
    out[out_mask == 0] = 255 #makes nuclei white on the black background
    # cv2.imwrite(outp + 'Blur_Coloc_' + filename_h, out)
    plt.imshow(out)
    # cv2.imwrite('/home/atte/Documents/PD_images//batch8_retry/Deconvolved_ims/25/COLOC_h.png', out)
    out = cv2.bitwise_not(out)
    return(out)
    print('colocalised image created!')

# main_dir = '/home/atte/Desktop/Testing_coloc/Deconvolved_ims2'
main_dir = sys.argv[1]
# main_dir = '/home/atte/Documents/PD_images/batch8_retry/18/18/Deconvolved_ims'
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
#get all images that match the pattern
# matches_list = []

# for filename in all_ims_paths:
#     if 'WS' in filename:
#         print(filename)
file_pairs = {} #key: hunu_ws_th file, value: col1a1_th
print('STARTING...')
for f in all_ims_paths:
    # print(f)
    filename = os.path.basename(f)
    print(filename)
    #if 'Segm' in filename and 'hunu' in filename and 'WS' in filename:
    # if 'hunu' in filename and 'WS' in filename:
    if 'WS' in filename:
        # my_search_string = os.path.basename(f)
        print('filename: '+filename)
        for f2 in all_ims_paths:
            filename2 = os.path.basename(f2)
            print('filename2: '+filename2)
            if 'col1a1' in filename2 and 'TH' in filename2 and filename[:18] in filename2:
            # if my_search_string[:18] in os.path.basename(f2):
                file_pairs[f] = f2
                #print(f)
        #         im_name = f.split('/')[-1]
        #         print('im_name: ' + im_name)
        #         n = len(file_path.split('/')) #get number of elements in list created by splitting file path
        #         save_path = "/".join(file_path.split("/", 2)[:-1])  #save path is the same directory as where the file was found
                
        #         #im_id: the image index is the first number, the image specific id is the second (created after deconvolution to find the col1a1-hunu pairs), third is the dir
        #         im_id = "_".join(im_name.split("_",3)[:3])
        #         ids.append(im_id)
        #         # n = 18
        #         # im_id = [im_name[i:i+n] for i in range(0, len(im_name), n)] #extracts the animal id and the code of the image
        #         # im_id = im_id[0]
        
        # match_hunu_col1 = list(filter(lambda x: im_id in x, all_ims_paths))
        # matches_list.append(match_hunu_col1)

#Now we perform colocalisation:

for hunu, col1a1 in file_pairs.items():
    print('hunu: ' +hunu)
    filename = os.path.basename(hunu)
    filename = filename.split('.')[0]
    print('col1a1: ' +col1a1)
    coloc_im = colocalise(hunu,col1a1)
    # coloc_im = cv2.bitwise_not(coloc_im)

    cv2.imwrite(coloc_dir +'/' + filename + "_Coloc.png",coloc_im)
    # coloc_im = Image.fromarray(np.uint8(coloc_im * 255))
    # coloc_im.save(coloc_dir +'/' + filename + "_Coloc.png")
    print('SAVED :' + coloc_dir +'/' + filename + "_Coloc.png" )
print(coloc_dir)
print("ALL COLOCALISED")
