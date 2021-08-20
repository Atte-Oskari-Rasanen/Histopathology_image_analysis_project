#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 15:21:50 2021

@author: atte
"""

# import the necessary packages
import matplotlib
matplotlib.use('Agg')
import numpy as np
from numpy import asarray
import PIL
import argparse
import cv2
from skimage import measure, filters
from PIL import Image as im

from skimage.filters import threshold_otsu
import os
from PIL import Image
from matplotlib import pyplot as plt

matplotlib.use('Agg')


rootdir = '/home/inf-54-2020/experimental_cop/H_final/Images/'
outputdir = '/home/inf-54-2020/experimental_cop/H_final/Masks/'
#rootdir = '/home/atte/Documents/images_qupath2/H_final/'
#outputdir='/home/atte/Documents/images_qupath2/H_final/Masks2/'
print(rootdir)

#for path, dirs, files in os.walk(rootdir):
for f in os.listdir(rootdir):
    #print(files)
        
    print(f)
    f_name=f.rsplit('.', 1)[0]
    #print(f_name)
    img_path = rootdir + f
    n_path = outputdir + f 
    #if not Image.open(n_path):
     #   continue
    #else:
    # Create a mask using the DAPI image and binary thresholding at 25
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    img = np.array(img)
    thresh = threshold_otsu(img)
    binary = img > thresh
    
    #binary = [img, 0, th]
    #img = np.digitize(img, bin=np.array([ret]))
    
    #print(binary)
    #print(img)
    #img=img.convert("L") #grayscale
    #img = np.array(img)
    #print(img)
    #TH = filters.threshold_otsu(img)  #apply threshold, save it
    #thresh = threshold_otsu(img)
    #binary = img > TH
    #plt.imshow(binary)
    #img = im.fromarray(binary)
    
    #Save the thresholded image into an output (first transform into uint8 since pil needs this)
    img = Image.fromarray((binary * 255).astype(np.uint8))
    img.save(n_path, img)

    #plt.savefig(n_path, bbox_inches = 'tight', pad_inches = 0)
    print('saved!')
# =============================================================================
#             TH = filters.threshold_otsu(img)  #apply threshold, save it
#             
#             print(TH)
#             #th = cv.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#             pyplot.imshow(TH)
#             pyplot.savefig(n_path, bbox_inches='tight')
#             #cv2.imwrite(n_path, th)
# =============================================================================

# =============================================================================
#             thresh = threshold_otsu(img)
#             binary = img > thresh
#             #print(binary)
#             binary = binary.astype(np.int8)
#             print(binary)


# =============================================================================
#     print(files)
#      if file in files:
#           print('found %s' % os.path.join(path, file))
# 
# for subdir, dirs, files in os.walk(rootdir):
#     for name in dirs:
#         
#        print(os.path.join(rootdir, name))
#         if file.endswith(".png"):
#             f_name=file.rsplit('.', 1)[0]
#             n_path = outputdir + f_name + '/' 
#             #if not Image.open(n_path):
#              #   continue
#             #else:
#             img = Image.open(n_path)
#             print(img)
#             img=img.convert("L") #grayscale
#             img = np.array(img)
#             thresh = threshold_otsu(img)
#             binary = img > thresh
#             binary = binary.astype(np.uint8)
# 
#             cv2.imwrite(n_path, binary)
# 
#             #os.path.join(subdir, file)
# 
# #Save the thresholded image into an output (first transform into uint8 since pil needs this)
# # =============================================================================
# =============================================================================
# for file in os.listdir(path):
# 
#      if file.endswith(".jpg") or file.endswith(".tif"): 
#         imagepath=path + "/" + file
#         
#         img=Image.open(imagepath)
#         img=img.convert("L") #grayscale
#         img = np.array(img)
#         thresh = threshold_otsu(img)
#         binary = img > thresh
#         mask = 'mask_'
#         #img = Image.fromarray((binary * 255).astype(np.uint8))
#         final_name = mask + file
#         cv2.imwrite(final_name, img)
#         print('files processed!')
# # =============================================================================
# =============================================================================
#         img = Image.fromarray((binary * 255).astype(np.uint8))
# 
#         print(img)
#         img = asarray(img)
#         print(type(img))
#         print('------')
#         #blur = cv.GaussianBlur(img,(5,5),0)
#         img = cv.threshold(img,0,255, cv.THRESH_BINARY + cv.THRESH_OTSU)
#         print(type(img))
#         img = np.asarray(img)
#         print(type(img))
#         img = Image.fromarray((img * 255).astype(np.uint8))
#         #pyplot.imshow(img)
#         pyplot.savefig(outpath+'Mask_' + file, transparent = True, bbox_inches='tight', pad_inches=0)
# 
# =============================================================================
# =============================================================================
#         img = color.rgb2gray(io.imread(imagepath))
#         img = asarray(img)
#         pyplot.set_axis_off()
# 
#         #img = load_img(imagepath)
#         thresh = threshold_otsu(img)
#         binary = img < thresh
#         pyplot.axis('off')
# 
#         #plt.imshow(binary)
#         img = Image.fromarray((binary * 255).astype(np.uint8))
#         pyplot.imshow(img)
#         pyplot.savefig(outpath+'_Mask.tif', transparent = True, bbox_inches='tight', pad_inches=0)
# 
# =============================================================================
