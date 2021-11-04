#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 21:43:28 2021

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

# Load image, grayscale, Otsu's threshold
h_path = '/home/atte/Documents/PD_images/batch6/DAB15/DAB_15s_hunu_segm.png' 
h_path = '/home/atte/Documents/PD_images/batch6/10ep_Alldat_kaggleDice_S_DAB_Hunu_15sec_512.png'
h_path = '/home/atte/Documents/PD_images/batch6/DAB15/u2_5ep_bs128_dice_DAB_15sec_512.png'
h_path ='/home/atte/Desktop/quick/U2_ep3_Alldat_bs128_dice_DAB_15sec_s736.png'
# h_path = '/home/inf-54-2020/experimental_cop/All_imgs_segm/u2_5ep_bs128_dice_DAB_15sec_512.png'
c_path = '/home/atte/Documents/PD_images/batch6/DAB15/DAB_15sec_col1a1.png'
c_path = '/home/atte/Desktop/quick/col1a1_DAB15sec.png'
outp = '/home/atte/Documents/PD_images/batch6/'

#for the images segmented with script U2.py need to invert colours
from skimage.morphology import disk
from scipy.ndimage.filters import gaussian_filter
from skimage import util 
import cv2
import numpy as np
import PIL
import os
import re
from watershed_hunu import *
from PIL import Image, ImageFilter
from skimage.filters import threshold_otsu, rank
import scandir
#go over the deconvolved folder, find folders that have Segmented in their names, enter the folder
#and apply hunu_ch_import_TH on it. Find the corresponding col1a1 image , threshold

def hunu_ch_import_TH(im_path, radius, sigma):
    img = cv2.imread(im_path,0)
    selem = disk(radius)
    im_blur = gaussian_filter(img, sigma=sigma)
    
    print(im_blur.shape)
    local_otsu = rank.otsu(im_blur, selem)
    binary = im_blur >= local_otsu
    
    print(binary.dtype)
    binary = binary.astype(np.uint8)


    return binary
#cv2.imwrite('/home/atte/Documents/PD_images/batch6/t.png', threshInv)

def col1a1_ch_import_TH(im_path):
    kernel = np.ones((5,5),np.uint8)

#image = cv2.imread(imagepath)
    im_gray = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
    thresh = threshold_otsu(im_gray)
    
    #add extra on top of otsu's thresholded value as otsu at times includes background noise
    thresh = thresh - 20
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #blurred = cv2.GaussianBlur(im_gray, (7, 7), 0)
    im_gray = Image.fromarray(im_gray)
    im_blur = im_gray.filter(ImageFilter.GaussianBlur(20))
    im_blur = np.asarray(im_blur)
    
    (T, threshInv) = cv2.threshold(im_blur, thresh, 255,
    	cv2.THRESH_BINARY_INV)
    threshInv = cv2.dilate(threshInv,kernel,iterations = 1)
    # np.invert(threshInv)

    return threshInv
#cv2.imwrite('/home/atte/Documents/PD_images/batch6/dab_binary_col1a1.png', threshInv)
def colocalise(hunu_im, col1a1_im):
    hunu_im = cv2.imread(hunu_im,0)
    col1a1_im = cv2.imread(col1a1_im,0)
    print(col1a1_im.shape)
    # cv2.imwrite('/home/atte/Documents/PD_images/batch6/col1a1.png', col1a1_im)
    cnts, _ = cv2.findContours(col1a1_im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #get the contours of the col1a1 
    #_ , contours, _ = cv2.findContours(threshInv,2,1)            
    contours = sorted(cnts, key=cv2.contourArea)            #get the largest contour
    
    out_mask = np.zeros_like(hunu_im)
    
    #use this when applying mask to the image of nuclei
    cv2.drawContours(out_mask, cnts, -1, 255, cv2.FILLED, 1)                                        
    
    
    out=hunu_im.copy()
    out[out_mask == 0] = 255 #makes nuclei white on the black background
    # cv2.imwrite(outp + 'Blur_Coloc_' + filename_h, out)
    return(out)
    print('colocalised image created!')

# h_path = sys.argv[0]
# c_path = sys.argv[1]
# outp = sys.argv[2]
# print(h_path)

# directory = sys.argv[1]
# patch_size = int(sys.argv[2])
# segm_model = sys.argv[3]
main_dir = './deconv'
main_dir = '/home/atte/Desktop/Testing_coloc/hunu_th'


# create the colocalisation when you colocalise the images since prior to colocalisation you need
# to find the matching files. create the directory within this same match condition
# create a colocalised folder under each animal id 
for root, subdirectories, files in scandir.walk(main_dir):
    print(subdirectories)
    for subdir in subdirectories:
        if not 'Coloc' in subdir:
            coloc_dir = main_dir + '/Coloc'
            try:
                os.mkdir(coloc_dir)
            except OSError:
                print ("Failed to create directory %s " % coloc_dir)
            else:
                print ("Succeeded at creating the directory %s " % coloc_dir)

#get all images in a list
segm_dirs = []
all_ims_paths = []
for (dirpath, dirnames, filenames) in os.walk(main_dir):
    all_ims_paths += [os.path.join(dirpath, file) for file in filenames]

print('all imgs paths:')
# print(all_ims_paths)
#get all images that match the pattern
for f in all_ims_paths:
    #print(f)
    im_name = f.split('/')[-1]
    # print(im_name)
    n = 18
    im_id = a = [im_name[i:i+n] for i in range(0, len(im_name), n)] #extracts the animal id and the code of the image
    im_id = im_id[0]
    match_hunu_col1 = list(filter(lambda x: im_id in x, all_ims_paths))
    # matches_list.append(match_hunu_col1)
    # print(im_id)


    #now you have matching image ids for col1a1, hunu and hunu_segm. you now go through 
    #the list containing all the images that were saved earlier to find the corresponding ones
    #and take the col1a1 and hunu_segm
    for file_path in match_hunu_col1:
        filename = os.path.basename(file_path)
        filename = filename.split('.')[0]
        print('file_path: ' + file_path)
        # print('file_name: ' + filename)

        # print(file_path)
        animal_id = file_path.split('_')[-3]
        if 'col1a1' in filename and not 'TH' in filename:
            # print(filename)
            #get filename
            print('col1a1 name: ' + filename)
            col1a1 = col1a1_ch_import_TH(file_path)
            splt_char = "/"
            # nth = 4
            # split_path = file_path.split('/')
            # th_path = '_'.join(split_path[:n]), '_'.join(split_path[n:])
            # print(th_path)
            # col1a1_th_path = th_path[1]
            n = len(file_path.split('/')) #get number of elements in list created by splitting file path
            save_path = "/".join(file_path.split("/", n)[:-1])  #save path is the same directory as where the file was found
            print(save_path)
            print('col1a1 shape:' + str(col1a1.shape))
            cv2.imwrite(save_path + '/' + filename + '_TH.png', col1a1)
            print('thresholded col1a1 saved at '+ save_path)


ids= []
segm_TH_dirs = []
all_ims_paths = []
for (dirpath, dirnames, filenames) in os.walk(main_dir):
    all_ims_paths += [os.path.join(dirpath, file) for file in filenames]
#get all images that match the pattern
# matches_list = []

file_pairs = {} #key: hunu_ws_th file, value: col1a1_th
for f in all_ims_paths:
    filename = os.path.basename(f)

    if 'Segm' in filename and 'hunu' in filename and 'TH_WS' in filename:
        my_search_string = os.path.basename(f)
        for f2 in all_ims_paths:
            filename2 = os.path.basename(f2)

            if 'col1a1' in filename2 and 'TH' in filename2 and filename[:18] in filename2:
            # if my_search_string[:18] in os.path.basename(f2):
                file_pairs[f] = f2
HunuWSTH_Col1a1_TH = [None, None] #save the matching files into the list

for hunu, col1a1 in file_pairs.items():
    print('hunu: ' +hunu)
    filename = os.path.basename(hunu)
    filename = filename.split('.')[0]
    print('col1a1: ' +col1a1)
    coloc_im = colocalise(hunu,col1a1)
    cv2.imwrite(coloc_dir +'/' + filename + "_Coloc.png",coloc_im)
    # coloc_im = Image.fromarray(np.uint8(coloc_im * 255))
    # coloc_im.save(coloc_dir +'/' + filename + "_Coloc.png")
    print('saved :' + coloc_dir +'/' + filename + "_Coloc.png" )
    

