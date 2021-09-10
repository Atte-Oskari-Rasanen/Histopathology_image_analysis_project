#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 16:01:01 2021

@author: atte
"""

from keras.utils import normalize
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from skimage import measure, color, io
import tensorflow as tf
from U_net_model import Unet
import os
seed = 42
np.random.seed = seed
import numpy as np
from tensorflow import keras
from tifffile import imsave
import ntpath
from Predict_indiv_img import Segment_img
IMG_HEIGHT = 512
IMG_WIDTH  = 512
IMG_CHANNELS = 3





cp_save_path = "/home/inf-54-2020/experimental_cop/scripts/kaggle_model_size512.h5"
model = keras.models.load_model(cp_save_path)

def get_model():
    return Unet(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

#Load the model and corresponding weights
model = get_model()
#model.load_weights('mitochondria_50_plus_100_epochs.hdf5') #Trained for 50 epochs and then additional 100
#model.load_weights(cp_save_path) #Trained for 50 epochs


save_path = "/home/inf-54-2020/experimental_cop/All_imgs_segm/"


#Load and process the test image - image that needs to be segmented. 
input_path = '/home/inf-54-2020/experimental_cop/Original_Images/Hu_D_30_min_10X.tif'
# test_img_norm = np.expand_dims(normalize(np.array(test_img), axis=1),2)
# test_img_norm=test_img_norm[:,:,0][:,:,None]
# test_img_input=np.expand_dims(test_img_norm, 0)

#Predict and threshold for values above 0.5 probability
segmented = Segment_img(input_path,model)
segmented = np.array(segmented)

#Load and process the test image - image that needs to be segmented. 
#test_img = cv2.imread('data/test_images/01-1_256.tif', 0)
#test_img = cv2.imread('/cephyr/NOBACKUP/groups/snic2021-23-496/Original_Images/Hu_D_30_min_10X.tif')
# test_img = cv2.imread('/home/inf-54-2020/experimental_cop/Original_Images/Hu_D_30_min_10X.tif')


# test_img_norm = np.expand_dims(normalize(np.array(test_img), axis=1),2)
# test_img_norm=test_img_norm[:,:,0][:,:,None]
# test_img_input=np.expand_dims(test_img_norm, 0)

# #Predict and threshold for values above 0.5 probability
# segmented = (model.predict(test_img_input)[0,:,:,0] > 0.05).astype(np.uint8)

plt.imsave(save_path + 'output_pre_instance.jpg', segmented, cmap='gray')

#print(test_img.shape)


########################################################
#####Watershed

img = cv2.imread(save_path +'output_pre_instance.jpg')  #Read as color (3 channels)
img_grey = img[:,:,0]

## transform the unet result to binary image
#Threshold image to binary using OTSU. ALl thresholded pixels will be set to 255
from skimage.morphology import disk
from skimage.filters import threshold_otsu, rank
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

radius = 5
selem = disk(radius)

thresh = rank.otsu(img_grey, selem)

#ret1, thresh = cv2.threshold(img_grey, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)


# Morphological operations to remove small noise - opening
#To remove holes we can use closing
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

#from skimage.segmentation import clear_border
#opening = clear_border(opening) #Remove edge touching grains. 
#Check the total regions found before and after applying this. 

#Now we know that the regions at the center of cells is for sure cells
#The region far away is background.
#We need to extract sure regions. For that we can use erode. 
#But we have cells touching, so erode alone will not work. 
#To separate touching objects, the best approach would be distance transform and then thresholding.

# let us start by identifying sure background area
# dilating pixes a few times increases cell boundary to background. 
# This way whatever is remaining for sure will be background. 
#The area in between sure background and foreground is our ambiguous area. 
#Watershed should find this area for us. 
sure_bg = cv2.dilate(opening,kernel,iterations=10)

# Finding sure foreground area using distance transform and thresholding
#intensities of the points inside the foreground regions are changed to 
#distance their respective distances from the closest 0 value (boundary).
#https://www.tutorialspoint.com/opencv/opencv_distance_transformation.htm
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)


#Let us threshold the dist transform by starting at 1/2 its max value.
ret2, sure_fg = cv2.threshold(dist_transform, 0.2*dist_transform.max(),255,0)

#Later you may realize that 0.2*max value may be better. Also try other values. 
#High value like 0.7 will drop some small mitochondria. 

# Unknown ambiguous region is nothing but bkground - foreground
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

#Now we create a marker and label the regions inside. 
# For sure regions, both foreground and background will be labeled with positive numbers.
# Unknown regions will be labeled 0. 
#For markers let us use ConnectedComponents. 
ret3, markers = cv2.connectedComponents(sure_fg)

#One problem rightnow is that the entire background pixels is given value 0.
#This means watershed considers this region as unknown.
#So let us add 10 to all labels so that sure background is not 0, but 10
markers = markers+10

# Now, mark the region of unknown with zero
markers[unknown==255] = 0
#plt.imshow(markers, cmap='gray')   #Look at the 3 distinct regions.

#Now we are ready for watershed filling. 
markers = cv2.watershed(img, markers)
#plt.imshow(markers, cmap='gray')
#The boundary region will be marked -1
#https://docs.opencv.org/3.3.1/d7/d1b/group__imgproc__misc.html#ga3267243e4d3f95165d55a618c65ac6e1

#Let us color boundaries in yellow. 
img[markers == -1] = [0,255,255]  

img2 = color.label2rgb(markers, bg_label=0)

# cv2.imshow('Overlay on original image', img)
# cv2.imshow('Colored Grains', img2)
# cv2.waitKey(0)

#Now, time to extract properties of detected cells
# regionprops function in skimage measure module calculates useful parameters for each object.

imagename = 'Hu_D_10x_30-min_test'
img2 = Image.fromarray((img2 * 255).astype(np.uint8))

#img2 = Image.fromarray(img2)
#print(type(img2))
img2.save(save_path + imagename + '_IS_WS.png')


props = measure.regionprops_table(markers, intensity_image=img_grey, 
                              properties=['label',
                                          'area', 'equivalent_diameter',
                                          'mean_intensity', 'solidity'])
    

import pandas as pd
df = pd.DataFrame(props)
df = df[df.mean_intensity > 100]  #Remove background or other regions that may be counted as objects

print(df.head())

#img = Image.fromarray(img2)
plt.imshow(segmented, cmap='gray')
plt.savefig('plot2_kaggledata.png')

print('Done!')

# plt.figure(figsize=(8, 8))
# plt.subplot(221)
# plt.title('Testing Image')
# plt.imshow(test_img, cmap='gray')
# plt.subplot(222)
# plt.title('Segmented Image')
# plt.imshow(segmented, cmap='gray')
# plt.show()

