#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 18:35:27 2021

@author: atte
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 18:35:27 2021

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
from scipy import ndimage as ndi
import os

import sys


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
#go over the deconvolved folder, find folders that have Segmented in their names, enter the folder
#and apply hunu_ch_import_TH on it. Find the corresponding col1a1 image , threshold
import PIL.ImageOps    

def crop(img):
    _,thresh = cv2.threshold(img,1,255,cv2.THRESH_BINARY)
    plt.imshow(thresh)
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)
    crop = img[y:y+h,x:x+w]
    return(crop)
def TH_local_otsu(img_p,radius, sigma):
    img = cv2.imread(img_p,0)
    selem = disk(radius)
    im_blur = gaussian_filter(img, sigma=sigma)
    # im_blur = cv2.medianBlur(im_blur, 3)
    print(im_blur.shape)
    local_otsu = rank.otsu(im_blur, selem)
    binary = im_blur >= local_otsu
    print(binary.dtype)
    binary = binary.astype(np.uint8)
    # binary = cv2.bitwise_not(binary)
    # plt.imshow(binary)
    binary = Image.fromarray(np.uint8(binary * 255))
    binary = PIL.ImageOps.invert(binary)

    # img.save(outp+ 'f_otsu_local_nogrids.png')
    # cv2.imwrite(outp + str(r) +'otsu_local.png', binary)
    return binary

print('starting the Stain_channels_postprocess script!')

#need to include an option for the user to select between which thresholding function to use with the nuclear marker channel (hunu) since
#at times the thresholding works better with local otsu rather than global one implemented in hunu_ch_import() 
def hunu_ch_import_TH(im_path):
    kernel = np.ones((5,5),np.uint8)

    img = cv2.imread(im_path,0)
    img = crop(img)
    thresh = threshold_otsu(img)

    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #blurred = cv2.GaussianBlur(im_gray, (7, 7), 0)
    im_gray = Image.fromarray(img)
    im_blur = im_gray.filter(ImageFilter.GaussianBlur(2))
    im_blur = np.asarray(im_blur)
    
    (T, threshInv) = cv2.threshold(im_blur, thresh, 255,
    	cv2.THRESH_BINARY_INV)


    return threshInv
#cv2.imwrite('/home/atte/Documents/PD_images/batch6/t.png', threshInv)

def col1a1_ch_import_TH(im_path):
    img = cv2.imread(im_path,0)
    thresh = threshold_otsu(img)
    #add extra on top of otsu's thresholded value as otsu at times includes background noise
    thresh = thresh #- thresh * 0.035   #need to remove a bit from the standard threshold and found this constant to be appropriate

    im_gray = Image.fromarray(img)
    im_blur = im_gray.filter(ImageFilter.GaussianBlur(5))
    im_blur = np.asarray(im_blur)
    
    (T, threshInv) = cv2.threshold(img, thresh, 255,
    	cv2.THRESH_BINARY_INV)
    threshInv = cv2.bitwise_not(threshInv)

    kernel = np.ones((5,5),np.uint8)
    threshInv = cv2.dilate(threshInv,kernel,iterations = 1)
    threshInv = cv2.bitwise_not(threshInv)
    threshInv = cv2.dilate(threshInv,kernel,iterations = 1)

    return threshInv
#cv2.imwrite('/home/atte/Documents/PD_images/batch6/dab_binary_col1a1.png', threshInv)
def colocalise(hunu_im, col1a1_im):

    col1a1_im = cv2.imread(col1a1_im,0)
    w,h=col1a1_im.shape
    hunu_im = cv2.imread(hunu_im,0)
    hunu_im = hunu_im.resize(w,h)
    print('col and hunu shapes:' + str(col1a1_im.shape) + str(hunu_im.shape))
    # cv2.imwrite('/home/atte/Documents/PD_images/batch6/col1a1.png', col1a1_im)
    cnts, _ = cv2.findContours(col1a1_im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #get the contours of the col1a1 
    #_ , contours, _ = cv2.findContours(threshInv,2,1)            
    contours = sorted(cnts, key=cv2.contourArea)            #get the largest contour
    out_mask = np.zeros_like(hunu_im)
    #draw contours of col1a1 image onto the hunu one

    cv2.drawContours(out_mask, cnts, -1, 255, cv2.FILLED, 1)                                        
    
    #cv2.drawContours(Img, cnts, -1, (0, 0, 255), 2) #-1 means draw all contours, red color, 2 is the width of contour line
    
    out=hunu_im.copy()
    out[out_mask == 0] = 255 #makes nuclei white on the black background
    # cv2.imwrite(outp + 'Blur_Coloc_' + filename_h, out)
    return(out)
    print('colocalised image created!')

main_dir = sys.argv[1]
print(main_dir)

# create the colocalisation folder into which you will save the colocalised files.
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

#from all images found get the ones with matching IDs. 
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
    #and take the col1a1 and hunu_segm and overlap them, only including the nuclei found within col1a1 contours. 
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

            col1a1= col1a1_ch_import_TH(file_path)
            # print('col1a1 shape: ' + str(col1a1.shape))
            splt_char = "/"

            n = len(file_path.split('/')) #get number of elements in list created by splitting file path
            save_path = "/".join(file_path.split("/", n)[:-1])  #save path is the same directory as where the file was found

            cv2.imwrite(save_path + '/' + filename + '_TH.png', col1a1)
            # col1a1.save(save_path + '/' + filename + '_TH.png')
            print('thresholded col1a1 saved at '+ save_path)

            # cv2.imwrite(save_path + '/' + filename + '_TH.png', col1a1)
            print('thresholded col1a1 saved at '+ save_path)
        if 'hunu' in filename and 'Segm' in filename and not 'TH' in filename:
            print('file_path = ' + filename)
            #get filename
            # print('hunu_segm name: ' + file_path)
            hunu = TH_local_otsu(file_path,30,5)
            n = len(file_path.split('/')) #get number of elements in list created by splitting file path
            save_path = "/".join(file_path.split("/", n)[:-1])  #save path is the same directory as where the file was found
            print(save_path)

            hunu.save(save_path + '/' + filename + '_TH.png')
            print('thresholded hunu saved at '+ save_path)
