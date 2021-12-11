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
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage as ndi
import os
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

import sys


#for the images segmented with script U2.py need to invert colours
from skimage.morphology import disk
from scipy.ndimage.filters import gaussian_filter
from skimage import util 
import cv2
import numpy as np
import PIL
import os
import re
from PIL import Image, ImageFilter
from skimage.filters import threshold_otsu, rank
import scandir
#go over the deconvolved folder, find folders that have Segmented in their names, enter the folder
#and apply hunu_ch_import_TH on it. Find the corresponding col1a1 image , threshold
import PIL.ImageOps    

def crop_image(image, angle):
    h, w = image.shape
    tan_a = abs(np.tan(angle * np.pi / 180))
    b = int(tan_a / (1 - tan_a ** 2) * (h - w * tan_a))
    d = int(tan_a / (1 - tan_a ** 2) * (w - h * tan_a))
    return image[d:h - d, b:w - b]

def TH_local_otsu(img_p,radius, sigma):
    img = cv2.imread(img_p,0)
    selem = disk(radius)
    im_blur = gaussian_filter(img, sigma=sigma)
    # im_blur = cv2.medianBlur(im_blur, 3)
    print(im_blur.shape)
    local_otsu = rank.otsu(im_blur, selem)
    binary = im_blur >= local_otsu
    # Creating kernel
    # kernel = np.ones((5, 5), np.uint8)
  
# Using cv2.erode() method 
    # binary = cv2.erode(binary, kernel) 

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
def hunu_ch_import_TH(im_path):
    kernel = np.ones((5,5),np.uint8)

    img = cv2.imread(im_path,0)
    img = crop_image(img,2)
    thresh = threshold_otsu(img)
    #add extra on top of otsu's thresholded value as otsu at times includes background noise
    thresh = thresh # - thresh * 0.035   #need to remove a bit from the standard threshold and found this constant to be appropriate
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #blurred = cv2.GaussianBlur(im_gray, (7, 7), 0)
    im_gray = Image.fromarray(img)
    im_blur = im_gray.filter(ImageFilter.GaussianBlur(2))
    im_blur = np.asarray(im_blur)
    
    (T, threshInv) = cv2.threshold(im_blur, thresh, 255,
    	cv2.THRESH_BINARY_INV)
    # threshInv = cv2.dilate(threshInv,kernel,iterations = 1)

    # threshInv = cv2.bitwise_not(threshInv)

    # binary = cv2.bitwise_not(binary) #invert colours so that cells are white and background is black
    # img = np.invert(binary)
    # img = util.invert(binary)

    # img = Image.fromarray(np.uint8(binary * 255))

    # img.save(outp + 'TEST_'+ filename_h)

    return threshInv
#cv2.imwrite('/home/atte/Documents/PD_images/batch6/t.png', threshInv)

def col1a1_ch_import_TH(im_path):
    img = cv2.imread(im_path,0)
    img = crop_image(img,2)
    thresh = threshold_otsu(img)
    #add extra on top of otsu's thresholded value as otsu at times includes background noise
    thresh = thresh #- thresh * 0.035   #need to remove a bit from the standard threshold and found this constant to be appropriate
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #blurred = cv2.GaussianBlur(im_gray, (7, 7), 0)
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
    # cv2.drawContours(hunu_im, contours, -1, (0, 0, 255), 2) #-1 means draw all contours, red color, 2 is the width of contour line
    #use this when applying mask to the image of nuclei
    cv2.drawContours(out_mask, cnts, -1, 255, cv2.FILLED, 1)                                        
    
    #cv2.drawContours(Img, cnts, -1, (0, 0, 255), 2) #-1 means draw all contours, red color, 2 is the width of contour line
    
    out=hunu_im.copy()
    out[out_mask == 0] = 255 #makes nuclei white on the black background
    # cv2.imwrite(outp + 'Blur_Coloc_' + filename_h, out)
    return(out)
    print('colocalised image created!')

# h_path = sys.argv[0]
# c_path = sys.argv[1]
# outp = sys.argv[2]
# print(h_path)
# for imagefile in os.listdir(h_path):  #to go through files in the specific directory
    #print(os.listdir(directory))
# imagepath=directory + "/" + imagefile
# if not imagefile.endswith('.tif') or imagefile.endswith('.jpg'): #exclude files not ending in .tif
#     continue
#print(imagepath)
# filename_h = os.path.basename(h_path)
# filename_c = os.path.basename(c_path)

    #print(imagename)
# hunu_im = hunu_ch_import_TH(h_path, 50, 10)
# hunu_im = np.asarray(hunu_im)
# col1a1_im = col1a1_ch_import_TH(c_path)
# cv2.imwrite(outp + 'Bin_COL1A1__'+ filename_c, col1a1_im)


# directory = sys.argv[1]
# patch_size = int(sys.argv[2])
# segm_model = sys.argv[3]
# main_dir = './deconv'
# main_dir = '/home/atte/Desktop/Testing_coloc/Deconvolved_ims2'
main_dir = '/home/atte/Documents/PD_images/batch8_retry/Deconvolved_ims'
# main_dir = sys.argv[1]
print(main_dir)

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
# matches_list = []
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

# print('match_hunu_col1: ')
# print(matches_list)

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
            # print('col1a1 file path: '+ file_path)
            # col1a1 = TH_local_otsu(file_path,30,5)
            col1a1= col1a1_ch_import_TH(file_path)
            # print('col1a1 shape: ' + str(col1a1.shape))
            splt_char = "/"
            # nth = 4
            # split_path = file_path.split('/')
            # th_path = '_'.join(split_path[:n]), '_'.join(split_path[n:])
            # print(th_path)
            # col1a1_th_path = th_path[1]
            n = len(file_path.split('/')) #get number of elements in list created by splitting file path
            save_path = "/".join(file_path.split("/", n)[:-1])  #save path is the same directory as where the file was found
            print(save_path)
            # temp = [x.start() for x in re.finditer(splt_char, file_path)]
            # res1 = file_path[0:temp[nth - 1]]
            # res2 = file_path[temp[nth - 1] + 1:]
            # split_path = (res1 + " " + res2).split(" ")
            # col1a1_th_path = col1a1_th_path[0] + '/'
            # print('col1a1_th_path: ' + col1a1_th_path)
            #coloc path corresponding to image:
            # col1a1 = Image.fromarray((col1a1 * 255).astype(np.uint8))
            # width, height = col1a1.size

            # left = width * 0.1
            # right = width - left
            # top = 0
            # bottom = height
            # im_final = col1a1.crop((left, top, right, bottom))
            # left = 0
            # right = width
            # top = height * 0.1
            # bottom = height - top
            # col1a1 = col1a1.crop((left, top, right, bottom))

            # print('col1a1 shape:' + str(col1a1.shape))
            # plt.imshow(col1a1)
            # col1a1_img.save(save_path + '/' + filename + '_TH.png')
            cv2.imwrite(save_path + '/' + filename + '_TH.png', col1a1)
            # col1a1.save(save_path + '/' + filename + '_TH.png')
            print('thresholded col1a1 saved at '+ save_path)

            # cv2.imwrite(save_path + '/' + filename + '_TH.png', col1a1)
            print('thresholded col1a1 saved at '+ save_path)
        if 'hunu' in filename and 'Segm' in filename and not 'TH' in filename:
            print('file_path = ' + filename)
            #get filename
            # print('hunu_segm name: ' + file_path)
            hunu = TH_local_otsu(file_path,30,1)
            # print(hunu[:1])
            # watershedded_hunu = watershedding(file_path, hunu)
            n = len(file_path.split('/')) #get number of elements in list created by splitting file path
            save_path = "/".join(file_path.split("/", n)[:-1])  #save path is the same directory as where the file was found
            print(save_path)
            # print('hunu_th_path: ' + hunu_th_path)
            # coloc_path = split_path + '/' + animal_id + '_Coloc'
            #coloc path corresponding to image:
            # print(coloc_path)
            # cv2.imwrite(save_path + '/' + filename + '_TH.png', hunu)
            hunu.save(save_path + '/' + filename + '_TH.png')
            print('thresholded hunu saved at '+ save_path)
