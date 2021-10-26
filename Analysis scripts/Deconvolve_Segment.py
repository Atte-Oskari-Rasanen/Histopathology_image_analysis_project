#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 21:03:11 2021

@author: atte
"""

import numpy as np
import matplotlib.pyplot as plt
from smooth_tiled_divisions import *

from skimage import data
from skimage.color import rgb2hed, hed2rgb
import cv2
import PIL
from PIL import Image
import sys
import os
import ntpath
import glob
import tensorflow
import keras

#the source for smooth_tiled_divisions scripts: 
# MIT License
# Copyright (c) 2017 Vooban Inc.
# Coded by: Guillaume Chevalier
# Source to original code and license:
#     https://github.com/Vooban/Smoothly-Blend-Image-Patches
#     https://github.com/Vooban/Smoothly-Blend-Image-Patches/blob/master/LICENSE

# from Threshold_Colocalise import *
# img_dir = sys.argv[1]


#from the command line enter the directory where the original images are, patch size for segmentation,
#model path and segm model name (e.g. if U net was used or VGG16 etc.). Deconvolved images main
#directory is created in the working directory


directory = '/home/atte/Documents/PD_images/test_decov/'
directory = sys.argv[1]
patch_size = int(sys.argv[2])
model_path = '/home/inf-54-2020/experimental_cop/scripts/Plots_Unet/Alldat_dice_ps736_bs128_ep3.h5'

segm_model = sys.argv[3]

#make a directory for deconvolved images:
main_dir = './Deconvolved_ims'



#generates random id used to identify the specific image that is deconvoluted into col1a1 and 
#hunu stains for later processing
import random
def im_id():
    seed = random.getrandbits(32)
    while True:
       yield seed
       seed += 1

uniq_id = im_id()


try:
    os.mkdir(main_dir)
except FileExistsError:
    print('Attempted to create a directory for deconvolved images but one already exists')
    pass
    
import scandir
for root, subdirectories, files in scandir.walk(directory):
    print(root)
    print(subdirectories)
    for subdir in subdirectories:
        print(subdir)
        animal_dir = main_dir + '/' + subdir
        #under Deconvolved_im/ make a subdir for each image id
        try:
            os.mkdir(animal_dir)
        except OSError:
            print ("Creation of the directory %s failed" % animal_dir)
        else:
            print ("Successfully created the directory %s " % animal_dir)

        hunu_ch_dir = subdir.rsplit('/')[-1] + '_hunu_ch'
        hunu_ch_dir = animal_dir + '/' + hunu_ch_dir
        col1a1_ch_dir = subdir.rsplit('/')[-1] + '_col1a1_ch'
        col1a1_ch_dir = animal_dir + '/' + col1a1_ch_dir
        print(hunu_ch_dir)
        print(col1a1_ch_dir)
        try:
            os.mkdir(hunu_ch_dir)
        except OSError:
            print ("Creation of the directory %s failed" % hunu_ch_dir)
        else:
            print ("Successfully created the directory %s " % hunu_ch_dir)
        
        print(col1a1_ch_dir)
        try:
            os.mkdir(col1a1_ch_dir)
        except OSError:
            print ("Creation of the directory %s failed" % col1a1_ch_dir)
        else:
            print ("Successfully created the directory %s " % col1a1_ch_dir)
        subdir_path = root +'/' +subdir + '/'
        
        print(subdir_path)
        print('subdir files:')
        #print(glob.glob(subdir_path))
        print(subdir_path + str(os.listdir(subdir_path)))
        im_index = 0
        for file in os.listdir(subdir_path):
            # print('file path:' )
            # print(os.path.join(subdir, file))
            print('file:'+file)
            imagepath=root + '/'+subdir + "/" + file
            if '~' in imagepath:
                imagepath = imagepath.split('~')[0]
            print('imagepath: '+ imagepath)
            # if not imagefile.endswith('.tif') or imagefile.endswith('.jpg'): #exclude files not ending in .tif
            #     continue
            #print(imagepath)
            imagename=ntpath.basename(imagepath)#take the name of the file from the path and save it
            imagename = imagename.split('.')[0]
            id_hunu_col = next(uniq_id) # specific id for the corresponding hunu and its col1a1 image

            imagename = str(im_index) + '_' +str(id_hunu_col) +'_'+ subdir + '_' + str(patch_size)
            # print(imagename)
            #get the threshold
            ihc_rgb = cv2.imread(imagepath)
            # print(ihc_rgb.shape)
            # ihc_rgb = CLAHE(ihc_rgb)
            # Separate the stains from the IHC image
            ihc_hed = rgb2hed(ihc_rgb)
            # ihc_hed = clahe.apply(ihc_hed)
            # Create an RGB image for each of the stains
            null = np.zeros_like(ihc_hed[:, :, 0])
            col1a1_ch = hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1))
            # ihc_e = hed2rgb(np.stack((null, ihc_hed[:, :, 1], null), axis=-1))
            hunu_ch = hed2rgb(np.stack((null, null, ihc_hed[:, :, 2]), axis=-1))
            col1a1_ch = Image.fromarray((col1a1_ch * 255).astype(np.uint8))
            hunu_ch = Image.fromarray((hunu_ch * 255).astype(np.uint8))
            col1a1_ch.save(col1a1_ch_dir + '/' + imagename + '_col1a1.tif')
            print('saved to ' + col1a1_ch_dir)
            hunu_ch.save(hunu_ch_dir + '/' + imagename + '_hunu.tif')
            im_index += 1
print('Deconvolution done! Starting segmentation...')

#cp_save_path = '/home/inf-54-2020/experimental_cop/scripts/Plots_Unet/Full_Model_5ep_dice_diceloss_ps512_bs128_ep3_gray.h5'
#cp_save_path = '/home/inf-54-2020/experimental_cop/scripts/transfer_learning_vgg16_ep3.h5'
model_segm = keras.models.load_model(model_path, compile=False) #need to set compile as false since we are predicting, >


# subdirs = os.listdir(main_dir)
# for subdir in subdirs:
#     if not '_' in subdir:
#         segm_dir = main_dir + '/' + subdir + '/' + segm_model +'_'+ str(patch_size) + '_Segmented'

#         print('segmdir: '+ segm_dir)
#         try:
#             os.mkdir(segm_dir)
#         except OSError:
#             print ("Creation of the directory %s failed" % segm_dir)
#         else:
#             print ("Successfully created the directory %s " % segm_dir)
        
        
os.walk(directory)
[x[0] for x in os.walk(directory)]
for root, subdirectories, files in scandir.walk(main_dir):
    print(subdirectories)
    for subdir in subdirectories:
        print('subdir:' + subdir)

        #first create a subdir where you save the segmented image. This dir is found inside the corresponding image to be segmented
        # segm_dir = subdir + '_' + segm_model +'_'+ str(patch_size) + '_Segmented'
        # segm_dir = main_dir + '/' + subdir + '_' + segm_model +'_'+ str(patch_size) + '_Segmented'
        # segm_dir = main_dir + '/' + subdir + '/' + segm_model +'_'+ str(patch_size) + '_Segmented'

        # print('segmdir: '+ segm_dir)
        # try:
        #     os.mkdir(segm_dir)
        # except OSError:
        #     print ("Creation of the directory %s failed" % segm_dir)
        # else:
        #     print ("Successfully created the directory %s " % segm_dir)
        
        # print('Number of files to process:' + file_count)

        subdir_path = root +'/' +subdir + '/'
        
        print(subdir_path)
        #print(glob.glob(subdir_path))
        for imagefile in os.listdir(subdir_path):
            if imagefile.endswith('.tif'):
                if 'hunu' in imagefile:
                    imagepath=subdir_path + "/" + imagefile
                    img = cv2.imread(imagepath)
        
                    imagename=ntpath.basename(imagepath)#take the name of the file from the path and save it
        #print(imagename)
        #get the threshold
                    #apply predict_img_with_smooth_windowing function which segments the image using the trained model
                    #supplied by cutting the image into patches of the appointed size, segmenting them, then combining 
                    #the patches
                    img_segm_grids_removed = predict_img_with_smooth_windowing(img,
                        window_size=patch_size,
                        subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
                        nb_classes=1,
                        pred_func=(
                            lambda img_batch_subdiv: model_segm.predict((img_batch_subdiv))
                        )
                    )
                    img_segm_grids_removed = np.squeeze(img_segm_grids_removed, axis = 2)
                    im_final = Image.fromarray((img_segm_grids_removed * 255).astype(np.uint8))
                    im_final.save(subdir_path + imagename + str(patch_size) + '_Segm.tif')
                else:
                    continue
            print('done!')



