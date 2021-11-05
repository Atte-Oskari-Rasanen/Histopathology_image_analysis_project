#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 12:49:53 2021

@author: atte
"""
import shutil

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
# import tensorflow
# import keras

#Python script for renaming and reorganising the deconvolved images performed by ImageJ and then
#segmenting them.

directory = '/home/atte/Documents/PD_images/test_decov/'
directory = sys.argv[1]
# patch_size = int(sys.argv[2])
# model_path = '/home/inf-54-2020/experimental_cop/scripts/Plots_Unet/Alldat_dice_ps736_bs128_ep3.h5'

# segm_model = sys.argv[3]

#make a directory for deconvolved images:
main_dir = './Deconvolved_ims'

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

a = list(range(1,10))
ints = list(map(str,a))
#Create the directory structure: ./Deconvolved_ims -> Animal ID -> col1a1_dir + hunu_dir
import scandir
for root, subdirectories, files in scandir.walk(directory):
    if 'Deconvolved_ims' in subdirectories:
        subdirectories.remove('Deconvolved_ims')


    print('initial root:' + root)
    print('initial subdir:' + str(subdirectories))
    for subdir in subdirectories:
        if any(integ in subdir for integ in ints): #check that the dir name contains an int
            print(subdir)
            animal_dir = main_dir + '/' + subdir
            #under Deconvolved_im/ make a subdir for each image id
            try:
                os.mkdir(animal_dir)
            except OSError:
                print ("Creation of the directory %s failed" % animal_dir)
            # else:
            #     print ("Successfully created the directory %s " % animal_dir)
    
            hunu_ch_dir = subdir.rsplit('/')[-1] + '_hunu_ch'
            hunu_ch_dir = animal_dir + '/' + hunu_ch_dir
            col1a1_ch_dir = subdir.rsplit('/')[-1] + '_col1a1_ch'
            col1a1_ch_dir = animal_dir + '/' + col1a1_ch_dir
            # print(hunu_ch_dir)
            # print(col1a1_ch_dir)
            try:
                os.mkdir(hunu_ch_dir)
            except OSError:
                print ("Creation of the directory %s failed" % hunu_ch_dir)
            # else:
            #     print ("Successfully created the directory %s " % hunu_ch_dir)
            
            # print(col1a1_ch_dir)
            try:
                os.mkdir(col1a1_ch_dir)
            except OSError:
                print ("Creation of the directory %s failed" % col1a1_ch_dir)
            # else:
            #     print ("Successfully created the directory %s " % col1a1_ch_dir)
            
            subdir_path = './' +subdir + '/'
            
            print('root: '+root)
            print('subdir'+subdir)

            print('subdir files:')
            #print(glob.glob(subdir_path))
            # subdir_path_to_orig_ims = root + '/' + subdir 
            # print(str(os.listdir(subdir_path)))
            
            #now you have created a directory under Deconvolved_ims with specific animal ID along with subdirs for 
            #hunu and col1a1. Now need to transfer the files from the original location to here.
            im_index = 0
            try:
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
        
                    imagename = str(im_index) + '_' +str(id_hunu_col) +'_'+ subdir + '_' #+ str(patch_size)
                    # print(imagename)
                    #get the threshold
                    
                    if 'col1a1' in file:
                        print('file col1: ' + file)
                        # Set the directory path where the file will be moved
                        destination_path = col1a1_ch_dir + '/' + file
                        try:
                            new_location = shutil.copyfile(imagepath, destination_path)
                        # new_location = shutil.move(imagepath, destination_path)
                            print("The %s is moved to the location, %s" %(file, new_location))
                            print('saved to ' + col1a1_ch_dir)
                        except FileNotFoundError:
                            pass
                    if 'hunu' in file:
                        print('file hunu: ' + file)
                        destination_path = hunu_ch_dir + '/' + file
                        try:
                            new_location = shutil.copyfile(imagepath, destination_path)
                            # new_location = shutil.move(imagepath, hunu_ch_dir)
                            print("The %s is moved to the location, %s" %(imagepath, new_location))
                        except FileNotFoundError:
                            pass
    
                    im_index += 1
            except FileNotFoundError:
                pass
        else:
            continue
print('Files reorganised! Starting segmentation...')

#The deconvolved images by ImageJ are in the directory of the specific animal. Take the images from
#here and transfer them to the right location inside Deconvolved_ims/. 



cp_save_path = '/home/inf-54-2020/experimental_cop/scripts/Plots_Unet/Full_Model_5ep_dice_diceloss_ps512_bs128_ep3_gray.h5'
cp_save_path = '/home/inf-54-2020/experimental_cop/scripts/transfer_learning_vgg16_ep3.h5'
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
        
        
# os.walk(directory)
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
                    width, height = im_final.size
                    # Setting the points for cropped image
                    left = width * 0.1
                    right = width - left
                    top = 0
                    bottom = height
                    im_final = im_final.crop((left, top, right, bottom))
                    left = 0
                    right = width
                    top = height * 0.1
                    bottom = height - top
                    im_final = im_final.crop((left, top, right, bottom))
                    im_final 
    
    # top = height / 4
    # right = 164
    # bottom = 3 * height / 4
    
    # Cropped image of above dimension
    # (It will not change original image)
    # img_segm_cropped = img_segm.crop((left, top, right, bottom))

                    im_final_crop = crop_borders(im_final)
                    im_final.save(subdir_path + imagename + str(patch_size) + '_Segm.tif')
                else:
                    continue
            print('done!')


