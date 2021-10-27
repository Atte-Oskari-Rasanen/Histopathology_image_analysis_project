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
# c_path = '/home/inf-54-2020/experimental_cop/batch6/DAB_15sec_col1a1.png'

# h_path = '/home/atte/Documents/PD_images/batch6/DAB30/DAB_30s_hunu_segm.png'
# c_path = '/home/atte/Documents/PD_images/batch6/DAB30/DAB_30sec_col1a1.png'

# h_path = '/home/atte/Documents/PD_images/batch6/DAB120/DAB_120s_hunu_segm.png'
# c_path = '/home/atte/Documents/PD_images/batch6/DAB120/DAB_120sec_col1a1.png'

# h_path = '/home/inf-54-2020/experimental_cop/All_imgs_segm/segm_batch6/DAB_15s_D_segm.png'
# c_path = '/home/inf-54-2020/experimental_cop/batch6/DAB_15sec_col1a1.png'
# outp = '/home/inf-54-2020/experimental_cop/batch6/batch6_coloc/'
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
#and apply hunu_ch_import_TH on it. Find the corresponding col1a1 image 

def hunu_ch_import_TH(im_path, radius, sigma):
    img = cv2.imread(im_path,0)
    selem = disk(radius)
    im_blur = gaussian_filter(img, sigma=sigma)
    
    print(im_blur.shape)
    local_otsu = rank.otsu(im_blur, selem)
    binary = im_blur >= local_otsu
    
    print(binary.dtype)
    binary = binary.astype(np.uint8)
    # binary = cv2.bitwise_not(binary) #invert colours so that cells are white and background is black
    # img = np.invert(binary)
    # img = util.invert(binary)

    # img = Image.fromarray(np.uint8(binary * 255))

    # img.save(outp + 'TEST_'+ filename_h)

    return img
#cv2.imwrite('/home/atte/Documents/PD_images/batch6/t.png', threshInv)

def col1a1_ch_import_TH(im_path):
    kernel = np.ones((5,5),np.uint8)

#image = cv2.imread(imagepath)
    im_gray = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
    thresh = threshold_otsu(im_gray)
    
    #add extra on top of otsu's thresholded value as otsu at times includes background noise
    thresh = thresh - 40
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
main_dir = './Deconvolved_ims'

segm_dirs = []
all_ims_paths = []
for (dirpath, dirnames, filenames) in os.walk(main_dir):
    all_ims_paths += [os.path.join(dirpath, file) for file in filenames]

print('all imgs paths:')
print(all_ims_paths)

#create a colocalised folder under each animal id 
for root, subdirectories, files in scandir.walk(main_dir):
    print(subdirectories)
    for subdir in subdirectories:
        coloc_dir = subdir + '/' + 'Coloc'
        try:
            os.mkdir(coloc_dir)
        except OSError:
            print ("Failed to create directory %s " % coloc_dir)
        else:
            print ("Succeeded at creating the directory %s " % coloc_dir)


for f in all_ims_paths:
    im_id = f.split('_')[1]
    match_hunu_col1 = list(filter(lambda x: im_id in x, all_ims_paths))
    # print('match_hunu_col1: ')
    # print(match_hunu_col1)

#now you have matching image ids for col1a1, hunu and hunu_segm. you now go through 
#the list containing all the images that were saved earlier to find the corresponding ones
#and take the col1a1 and hunu_segm
for fname in match_hunu_col1:
    # print(fname)
    animal_id = fname.split('_')[-3]
    if 'col1a1' in fname:
        print(fname)
        #get filename
        f = os.path.basename(fname)
        print('col1a1 name: ' + fname)
        col1a1 = col1a1_ch_import_TH(fname)
        splt_char = "_"
        nth = 4
        temp = [x.start() for x in re.finditer(splt_char, fname)]
        res1 = fname[0:temp[nth - 1]]
        res2 = fname[temp[nth - 1] + 1:]
        split_path = (res1 + " " + res2).split(" ")
        col1a1_th_path = split_path[0] + '/'
        # print('col1a1_th_path: ' + col1a1_th_path)
        #coloc path corresponding to image:
        cv2.imwrite(col1a1_th_path + 'TH_' + f, col1a1)
        # print('thresholded col1a1 saved at '+ col1a1_th_path)

    if 'Segm' in fname and 'hunu' in fname:
        print(fname)
        #get filename
        orig_im = os.path.basename(fname)
        print('hunu_segm name: ' + fname)
        hunu = hunu_ch_import_TH(fname, 50, 15)
        watershedded_hunu = watershedding(fname, hunu)
        splt_char = "/"
        nth = 4
        temp = [x.start() for x in re.finditer(splt_char, fname)]
        res1 = fname[0:temp[nth - 1]]
        res2 = fname[temp[nth - 1] + 1:]
        split_path = (res1 + " " + res2).split(" ")
        hunu_th_path = split_path[0] + '/'
        print('hunu_th_path: ' + hunu_th_path)
        # coloc_path = split_path + '/' + animal_id + '_Coloc'
        #coloc path corresponding to image:
        # print(coloc_path)
        watershedded_hunu.save(hunu_th_path + 'TH_WS_' + orig_im)
        print('thresholded hunu saved at '+ hunu_th_path)

#now both col1a1 and hunu channels have been postprocessed and thresholded. 
#the next stage is to get the colocalised image 

#     #now we have the thresholded images, next we apply colocalisation and save the images
#     coloc = colocalise(hunu, col1a1)
#     cv2.imwrite(outp + 'Blur_Coloc_' + filename_h, out)

# for root, subdirectories, files in scandir.walk(directory):
#     for subdir in subdirectories:
#         if 'Segmented' in subdir:
#             print('subdir segmented: ' + subdir)
#             #first create a subdir where you save the binary images. This dir is found inside the corresponding image to be segmented
#             bin_dir = subdir.rsplit('/')[-1] + '_Binary'
#             bin_dir = directory + subdir + '/' + segm_dir
#             try:
#                 os.mkdir(bin_dir)
#             except OSError:
#                 print ("Creation of the directory %s failed" % segm_dir)
#             else:
#                 print ("Successfully created the directory %s " % segm_dir)
            
#             # print('Number of files to process:' + file_count)

#             subdir_path = root +'/' +subdir + '/'
#             print(subdir_path)
            
# #iterate over subdirs, save file names to list. f
#             # segm_dirs = segm_dirs.append(subdir)
#             #print(glob.glob(subdir_path))
#             for imagefile in os.listdir(subdir_path):
                
#                 if 'col1a1' in subdir_path:
#                     print('found col1a1 in path: ' + subdir_path)
#                     col1a1_im = col1a1_ch_import_TH(c_path)
#                 else:
                        
#                 if imagefile.endswith('.tif'):
#                     imagepath=subdir_path + "/" + imagefile
#                     img = cv2.imread(imagepath)
#                     hunu_im = hunu_ch_import_TH(h_path, 50, 10)
    
#                     imagename=ntpath.basename(imagepath)#take the name of the file from the path and save it
    
#                     img_segm_grids_removed = np.squeeze(img_segm_grids_removed, axis = 2)
#                     im_final = Image.fromarray((img_segm_grids_removed * 255).astype(np.uint8))
#                     im_final_name = segm_dir + '/' + imagename + str(patch_size) + '_Bin.png'
#                     im_final.save(im_final_name)
#                     all_ims_paths.append(im_final_name)
#                 else:
#                     continue
#             print('done!')


# out=col1a1_im.copy()

