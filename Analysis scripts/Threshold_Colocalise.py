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
    # binary = cv2.bitwise_not(binary) #invert colours so that cells are white and background is black
    # img = np.invert(binary)
    # img = util.invert(binary)

    # img = Image.fromarray(np.uint8(binary * 255))

    # img.save(outp + 'TEST_'+ filename_h)

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
            # temp = [x.start() for x in re.finditer(splt_char, file_path)]
            # res1 = file_path[0:temp[nth - 1]]
            # res2 = file_path[temp[nth - 1] + 1:]
            # split_path = (res1 + " " + res2).split(" ")
            # col1a1_th_path = col1a1_th_path[0] + '/'
            # print('col1a1_th_path: ' + col1a1_th_path)
            #coloc path corresponding to image:
            print('col1a1 shape:' + str(col1a1.shape))
            # plt.imshow(col1a1)
            # col1a1_img = Image.fromarray(np.uint8(col1a1 * 255))
            # col1a1_img.save(save_path + '/' + filename + '_TH.png')
            cv2.imwrite(save_path + '/' + filename + '_TH.png', col1a1)
            print('thresholded col1a1 saved at '+ save_path)

        # if 'Segm' in file_path and 'hunu' in filename:
        #     print('file_path = ' + filename)
        #     #get filename
        #     # print('hunu_segm name: ' + file_path)
        #     hunu = hunu_ch_import_TH(file_path, 30, 15)
        #     print(hunu[:1])
        #     watershedded_hunu = watershedding(file_path, hunu)
        #     splt_char = "/"
        #     nth = 4
        #     temp = [x.start() for x in re.finditer(splt_char, file_path)]
        #     res1 = file_path[0:temp[nth - 1]]
        #     res2 = file_path[temp[nth - 1] + 1:]
        #     split_path = (res1 + " " + res2).split(" ")
        #     hunu_th_path = split_path[0] + '/'
        #     # print('hunu_th_path: ' + hunu_th_path)
        #     # coloc_path = split_path + '/' + animal_id + '_Coloc'
        #     #coloc path corresponding to image:
        #     # print(coloc_path)
        #     watershedded_hunu.save(hunu_th_path + 'TH_WS_' + filename)
        #     print('thresholded hunu saved at '+ hunu_th_path)

# coloc_dir = main_dir + '/Coloc'
# try:
#     os.mkdir(coloc_dir)
# except OSError:
#     print ("Failed to create directory %s " % coloc_dir)
# else:
#     print ("Succeeded at creating the directory %s " % coloc_dir)
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
    

# for file_path1 in match_hunu_col1:
#     filename_suffix = os.path.basename(file_path)
#     filename= filename_suffix.split('.')[0]
#     print(filename)

#     # print(file_path)
#     # animal_id = filename.split('_')[1]
#     save_path = "/".join(file_path.split("/", 2)[:-1])  #save path is the same directory as where the file was found

# #im_id: the image index is the first number, the image specific id is the second (created after deconvolution to find the col1a1-hunu pairs), third is the dir
#     im_id1 = "_".join(filename.split("_",3)[:3])
#     ids.append(im_id)
#     for file_path2 in match_hunu_col1:
#         filename_suffix = os.path.basename(file_path2)
#         filename= filename_suffix.split('.')[0]
#         print(filename)
        
#         # print(file_path)
#         # animal_id = filename.split('_')[1]
    
#     #im_id: the image index is the first number, the image specific id is the second (created after deconvolution to find the col1a1-hunu pairs), third is the dir
#         im_id2 = "_".join(filename.split("_",3)[:3])
#         # ids.append(im_id)
#     if im_id1 == im_id2:
#         # print('animal_id after splitting:' + str(im_id))
#         if 'col1a1' in filename and 'TH' in filename:
#             # print(filename)
#             #get filename
#             HunuWSTH_Col1a1_TH[1] = file_path
#             print('col1a1 filepath name: ' + file_path)
#             print(len(HunuWSTH_Col1a1_TH))
    
#         if 'Segm' in file_path and 'hunu' in filename and 'TH_WS' in filename:
#             print('hunu file_path = ' + file_path)
#             HunuWSTH_Col1a1_TH[0] = file_path
        
#         n = len(HunuWSTH_Col1a1_TH)
    
# for i in range(n):
    
# #run the colocalisation function by taking the files from the list
#     coloc_im = colocalise(HunuWSTH_Col1a1_TH[i],HunuWSTH_Col1a1_TH[i+1])
#     coloc_im = Image.fromarray(np.uint8(coloc_im * 255))

#     coloc_im.save(coloc_dir +'/' + filename + "_Coloc.png")
    


#save the location of all the files into a list (contains the file name and the abs.path). 
#go over the files, find the one which has names 'hunu' and 'TH_WS' in them, get the image, find
#the corresponding file which has the name 'col1a1' and 'TH'. To find the corresponindg ones base it
#on the given image id. then apply colocalise function and save the result to their own colocalise dir.


# segm_TH_dirs = []
# all_ims_paths = []

# for (dirpath, dirnames, filenames) in os.walk(main_dir):
#     all_ims_paths += [os.path.join(dirpath, file) for file in filenames]
# #get all images that match the pattern
# # matches_list = []
# for f in all_ims_paths:
#     #print(f)
#     im_name = f.split('/')[-1]
#     # print(im_name)
#     n = 18
#     im_id = a = [im_name[i:i+n] for i in range(0, len(im_name), n)] #extracts the animal id and the code of the image
#     im_id = im_id[0]
#     match_hunu_col1 = list(filter(lambda x: im_id in x, all_ims_paths))
#     # matches_list.append(match_hunu_col1)
#     for file_path in match_hunu_col1:
#         filename = os.path.basename(file_path)
#         print(filename)



# #Quantification

# h_c_list = []
# coloc_list = []
# unit_list = []
# filenames = []
# from skimage.filters import threshold_otsu, rank

# import scandir
# for root, subdirectories, files in scandir.walk(main_dir):
#     for subdir in subdirectories:
#         if 'Coloc' in subdir:
#             for file in files:
#                 if 'Segm' in file and 'WS_TH' in file:
#                     file = file.split('.')[0]
#                     print(file)
#                     filenames.append(file)
#                     im_path = root + '/' + subdir + file
#                     im_gray = cv2.imread(im_path)
#                     #count hunu cells within col1a1
#                     labels_coloc = measure.label(coloc_cells) #create a binary format of the numpy array (object or background)
#                     cell_count_coloc=labels_coloc.max() #count the number of objects
#                     coloc_list.append(cell_count_coloc)
#                     filename_h = os.path.basename(im_path)
#                     labels_h = measure.label(hunu_cells) #create a binary format of the numpy array (object or background)
#                     cell_count_h=labels_h.max() #count the number of objects
#                     coloc_list.append(cell_count_h)
#                     filenames.append(filename)
#         if 'hunu' in subdir:
#             for file in files:
#                 if 'Segm' in file and 'WS_TH' in file:
#                     im_path = root + '/' + subdir + file
#                     im_gray = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
#                     thresh = threshold_otsu(im_gray)
#                     filename_h = os.path.basename(im_path)
#                     labels_h = measure.label(hunu_cells) #create a binary format of the numpy array (object or background)
#                     cell_count_h=labels_h.max() #count the number of objects
#                     h_c_list.append(cell_count_h)
                    
                    
# df = pd.DataFrame(list(zip(filenames, h_c_list, coloc_list, unit_list)))
# df.columns = ["Filename", "Total # of HuNu cells", "# of Hunu cells colocalised with COL1A1", "#HUNU+COL1A1+ / HUNU+"]

# with open("/home/inf-54-2020/experimental_cop/scripts/Info.txt", "w") as out: #write the dataframe into an output file
# #write the dataframe into an output file
#     df.to_string(out, index=None)
#     print('output info file saved!')






#colocalisation part
# now both col1a1 and hunu channels have been postprocessed and thresholded. 
# the next stage is to get the colocalised image 

    # #now we have the thresholded images, next we apply colocalisation and save the images
    # coloc = colocalise(hunu, col1a1)
    # cv2.imwrite(outp + 'Blur_Coloc_' + filename_h, out)

# for root, subdirectories, files in scandir.walk(directory):
#     for subdir in subdirectories:
#         if 'TH_WS' in subdir:
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

