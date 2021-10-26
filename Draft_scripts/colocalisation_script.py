#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 12:02:39 2021

@author: atte
"""

#Libraries to import.
import pylab as plt
import cv2
from tifffile import imread
import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu
from scipy import ndimage
from scipy.ndimage.morphology import distance_transform_edt
from skimage.segmentation import flood_fill
from skimage.morphology import watershed
from findmaxima2d import find_maxima, find_local_maxima #install with pip E.g.: python3 -m pip install findmaxima2d



import matplotlib.pyplot as plt2
from PIL import Image

from skimage import data
from skimage.color import rgb2hed, hed2rgb

# Example IHC image
#ihc_rgb = imread('/home/atte/Documents/images_qupath2/3_cropped_20x.tif')
ihc_rgb = np.array(plt.imread('./images_qupath2/YZ004_NR_G2_#15_hCOL1A1_20x_(2).tif'), dtype='float64')[:, :, 0:3]

ihc_hed = rgb2hed(ihc_rgb)#separate stains


null = np.zeros_like(ihc_hed[:, :, 0])
ch0 = hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1))
ch1 = hed2rgb(np.stack((null, ihc_hed[:, :, 1], null), axis=-1))
ch2 = hed2rgb(np.stack((null, null, ihc_hed[:, :, 2]), axis=-1))

#plotting.
plt.figure(figsize=(32,32))
plt.subplot(1,2,1)
plt.imshow(ch0)
#plt.subplot(1,3,2)
#plt.imshow(ch1)
plt.subplot(1,2,2)
plt.imshow(ch2)

#Thersholding based on DAB
thresh = threshold_otsu(ch2)
binary = ch2 < thresh
plt.imshow(binary)

#Save the thresholded image into an output (first transform into uint8 since pil needs this)
img = Image.fromarray((binary * 255).astype(np.uint8))
img.save("THed.tif")

#watershed segmentation - find the bright areas (peaks) in the image 

#Colocalisation


ntol = 50 #the local pixel max
img_data = np.array(ch0)
print(img_data)
img_data = (img_data/np.max(img_data))*255 #scale 
#print('toka:')
#print(img_data)
#img_data = img_data[~img_data.columns.duplicated(keep='first')]
local_max = find_local_maxima(img_data) #get the local max
#store the outputs:
ych1,xch1, out = find_maxima(img_data, local_max.astype(np.uint8), ntol)
#plt.figure(figsize= (32,32))
#plt.plot(xch1,ych1,'ro')
#plt.imshow(img_data)
img = Image.fromarray((img_data * 255).astype(np.uint8))

img.save("THed_loc_maximas.tif")


ntol1 = 100
img_data = np.array(ch1)
img_data = (img_data/np.max(img_data))*255
local_max = find_local_maxima(img_data)
ych2,xch2, out = find_maxima(img_data, local_max.astype(np.uint8), ntol)
#plt.figure(figsize= (32,32))
#plt.plot(xch2,ych2,'ro')
#plt.imshow(img_data)

cell_ch1 = []
cell_ch2 = []

for i in range(0, num):
    cell_ch1.append([])
    cell_ch2.append([])
    bint = labels == i+1
    for yc1,xc1 in zip(ych1,xch1):
        if bint[yc1,xc1] == 1:
            cell_ch1[i].append([yc1,xc1])
    for yc2, xc2 in zip(ych2,xch2):
        if bint[yc2,xc2] == 1:
            cell_ch2[i].append([yc2,xc2]) 
df['pts_ch1'] = cell_ch1
df['pts_ch2'] = cell_ch2

coloc_num = []
for i in range(1,num+1):
    cell_ch1 = list(df[df['cell_num']==i]['pts_ch1'])[0]
    cell_ch2 = list(df[df['cell_num']==i]['pts_ch2'])[0]
    
    count =0
    for pts1 in cell_ch1:
        for pts2 in cell_ch2:
            dist = np.sqrt((pts1[0]-pts2[0])**2 + (pts1[1]-pts2[1])**2)
            if dist < 10:
                count +=1
    coloc_num.append(count)
df['coloc_num'] = coloc_num
plt.figure(figsize=(32,32))
plt.plot(xch1,ych1,'ro',alpha=0.5,markersize=25)
plt.plot(xch2,ych2,'ko',alpha=0.5,markersize=25)
plt.imshow(ch0)
