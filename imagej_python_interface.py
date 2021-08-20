#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 14:50:10 2021

@author: atte
"""
import pylab as plt
import imagej
import cv2
import skimage
from skimage import color
from skimage import io

import numpy as np
from skimage.filters import threshold_otsu
from PIL import Image
import imghdr

import os
import ntpath

#1. Import images - original, H, D, H_SW(segmented+watershed)
#2. Based on D image, threshold via otsu and get the coordinates of the outlines
#3. Apply these outlines to H and H_SW images
#4. Extract features of the H_SW cells - Area Perimeter Eccentricity Circularity 
# Class (WBC)
# pcj = Tcj / Ta   means prior prob of the class (human nucleus) equals to
# number of human nuclei of the training set divided by the total nuclei of the
# training set
directory = "~/Documents/images_qupath2/2nd_batch_process/Output/"

images_H = {} #dictionary with keys as image IDs and values contain lists consisting of path
#to the image as well as its respective otsu value
images_D = {} 


#To get hematoxylin
for imagefile in os.listdir(directory):  #to go through files in the specific directory
    paths = []
    #print(os.listdir(directory))
    imagepath=directory + "/" + imagefile   #create first of dic values, i.e the path
    if not imagefile.endswith('H_W_S.tif'): #exclude files not ending in .tif
        continue
    #print(imagepath)
    imagenameH=ntpath.basename(imagepath)#take the name of the file from the path and save it
        #append everything into a dictionary
    paths.append(imagepath)
    images_H[imagenameH]=paths

#To get DAB
for imagefile in os.listdir(directory):  #to go through files in the specific directory
    paths = []
    #print(os.listdir(directory))
    imagepath=directory + "/" + imagefile   #create first of dic values, i.e the path
    if not imagefile.endswith('_D.tif'): #exclude files not ending in .tif
        continue
    #print(imagepath)
    imagenameD=ntpath.basename(imagepath)#take the name of the file from the path and save it
        #append everything into a dictionary
    paths.append(imagepath)
    images_D[imagenameD]=paths

#Open the DAB image, threshold it, take the contour coordinates

for D in images_D.items():
    D_img = cv2.imread(D)
    th = threshold_otsu(D_img)
    binary = D_img < th

####testaus
#D_img = color.rgb2gray(io.imread("/home/atte/Documents/images_qupath2/2nd_batch_process/Output/YZ004 NR G2 #15 hCOL1A1 10x (1)_D.tif"))
H_img=cv2.imread("/home/atte/Documents/images_qupath2/2nd_batch_process/Output/YZ004 NR G2 #15 hCOL1A1 10x (1)_H_W_S.tif")
D_img = cv2.imread("/home/atte/Documents/images_qupath2/2nd_batch_process/10dabtest.tif")
#D_img.astype(np.uint8)
gray = cv2.cvtColor(D_img, cv2.COLOR_BGR2GRAY)
ht, wd, cc = D_img.shape

mask = cv2.imread("/home/atte/Documents/images_qupath2/2nd_batch_process/Output/YZ004 NR G2 #15 hCOL1A1 10x (1)_H_W_S.tif", cv2.IMREAD_GRAYSCALE)
mask = cv2.threshold(mask,0,255,cv2.THRESH_BINARY)[1]
mask = 255 - mask
hh, ww = mask.shape

# Now crop
mask = mask.transpose(2,0,1).reshape(-1,mask.shape[1])
(x, y) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
topx=topx
topy=topy
bottomx=bottomx
bottomy=bottomy
out = out[topx:bottomx,topy:bottomy]
print(out)


# =============================================================================
# Applying a low-pass blurring filter smooths edges and removes noise from an image.
# Blurring is often used as a first step before we perform Thresholding or before
# we find the Contours of an image
# =============================================================================

blurred = cv2.GaussianBlur(D_img, (7, 7), 0)
th = threshold_otsu(blurred)
binary = Image.fromarray((binary * 255).astype(np.uint8))

ret3,th3 = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# =============================================================================
# # apply Otsu's automatic thresholding which automatically determines
# # the best threshold value
# (T, threshInv) = cv2.threshold(blurred, 0, 255,
# 	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
# cv2.imshow("Threshold", threshInv)
# print("[INFO] otsu's thresholding value: {}".format(T))
# # visualize only the masked regions in the image
# masked = cv2.bitwise_and(D_img, D_img, mask=threshInv)
# cv2.imshow("Output", masked)
# cv2.waitKey(0)
# 
# =============================================================================
contours,hierachy=cv2.findContours(th3,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)#detect contours

idx=...
#mask = np.zeros_like(img) # Create mask where white is what we want, black otherwise
cv2.drawContours(H_img, contours, idx, 255, -1) # Draw filled contour in mask
i = 0
for c in contours:
    # get the bounding rect
    x, y, w, h = cv2.boundingRect(c)
    # to save the images
    cv2.imwrite('img_{}.jpg'.format(i), D_img[y:y+h,x:x+w])
    i += 1

####################

th = threshold_otsu(D_img)
binary = D_img < th
D_img_th = Image.fromarray((binary * 255).astype(np.uint8))
plt.imshow(D_img)
#plt.imshow(img)
#gray=cv2.cvtColor(binary,BGR2GRAY)#gray scale
contours, hierarchy = cv2.findContours(masked, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

ret,threshold=cv2.threshold(binary , 2 , 255 , cv2.THRESH_BINARY+cv2.THRESH_OTSU)
contours,hierachy=cv2.findContours(threshold,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)#detect contours
i = 0
for c in contours:
    # get the bounding rect
    x, y, w, h = cv2.boundingRect(c)
    # to save the images
    cv2.imwrite('img_{}.jpg'.format(i), image[y:y+h,x:x+w])
    i += 1

    mask = np.zeros_like(D_img) # Create mask where white is what we want, black otherwise
    cv2.drawContours(mask,  contours,  0,  (0 ,0,255),1)
    out = np.zeros_like(image) # Extract out the object and place into output image
    out[mask == 255] = image[mask == 255]

    img.save("THed.tif")

###Get the coordinates of the DAB channel 
# Load image

# Convert to grayscale and threshold
im = imread('THed.tif')


# convert to RGB
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
# convert to grayscale
gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
# =============================================================================
# # create a binary thresholded image
_, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)



# =============================================================================
# #Naive Bayesian
# # Import packages
# from sklearn.naive_bayes import GaussianNB
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns; sns.set()
# 
# # Import data
# training = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/iris_train.csv')
# test = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/iris_test.csv')
# 
# 
# # Create the X, Y, Training and Test
# xtrain = training.drop('Species', axis=1)
# ytrain = training.loc[:, 'Species']
# xtest = test.drop('Species', axis=1)
# ytest = test.loc[:, 'Species']
# 
# 
# # Init the Gaussian Classifier
# model = GaussianNB()
# 
# # Train the model 
# model.fit(xtrain, ytrain)
# 
# # Predict Output 
# pred = model.predict(xtest)
# 
# # Plot Confusion Matrix
# mat = confusion_matrix(pred, ytest)
# names = np.unique(pred)
# sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
#             xticklabels=names, yticklabels=names)
# plt.xlabel('Truth')
# plt.ylabel('Predicted')
# 
# =============================================================================
