#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 16:33:39 2021

@author: atte
"""

import pylab as plt
import cv2
from tifffile import imread
import numpy as np
import pandas as pd
import skimage
from skimage.filters import threshold_otsu
from scipy import ndimage
from scipy.ndimage.morphology import distance_transform_edt
from skimage.segmentation import flood_fill
from skimage.morphology import watershed
from findmaxima2d import find_maxima, find_local_maxima #install with pip E.g.: python3 -m pip install findmaxima2d

import imagej

import matplotlib.pyplot as plt2
from PIL import Image

from skimage.color import rgb2hed, hed2rgb

# Example IHC image
ihc_rgb = imread('/home/atte/Documents/images_qupath2/3_cropped_20x.tif')


for a in (ax[0], ax[1], ax[2]):
       a.axis('off')

ax[0].imshow(astro)
ax[0].set_title('Original Data')

ax[1].imshow(astro_noisy)
ax[1].set_title('Noisy data')

ax[2].imshow(deconvolved_RL, vmin=astro_noisy.min(), vmax=astro_noisy.max())
ax[2].set_title('Restoration using\nRichardson-Lucy')


fig.subplots_adjust(wspace=0.02, hspace=0.2,
                    top=0.9, bottom=0.05, left=0, right=1)
plt.show()
#ihc_rgb.show()
#ihc_rgb = np.array(plt.imread('./images_qupath2/YZ004_NR_G2_#15_hCOL1A1_20x_(2).tif'), dtype='float64')[:, :, 0:3]


ihc_hed = rgb2hed(ihc_rgb)#separate stains

null = np.zeros_like(ihc_hed[:, :, 0])
ch0 = hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1))
ch1 = hed2rgb(np.stack((null, ihc_hed[:, :, 1], null), axis=-1))
ch2 = hed2rgb(np.stack((null, null, ihc_hed[:, :, 2]), axis=-1))

#NNMF approach for deconvolution
imInput = skimage.io.imread('/home/atte/Documents/images_qupath2/3_cropped_20x.tif')[:, :, :3]

import histomicstk as htk

# create initial stain matrix
W = np.zeros((2,3))

# Compute stain matrix adaptively
sparsity_factor = 0.5

I_0 = 255
im_sda = htk.preprocessing.color_conversion.rgb_to_sda(imInput, I_0)
W_est = htk.preprocessing.color_deconvolution.separate_stains_xu_snmf(
    im_sda, W_init, sparsity_factor,
)

# perform sparse color deconvolution
imDeconvolved = htk.preprocessing.color_deconvolution.color_deconvolution(
    imInput,
    htk.preprocessing.color_deconvolution.complement_stain_matrix(W_est),
    I_0,
)

print('Estimated stain colors (in rows):', W_est.T, sep='\n')

# Display results
for i in 0, 1:
    plt.figure()
    plt.imshow(imDeconvolved.Stains[:, :, i])
    _ = plt.title(stains[i], fontsize=titlesize)

#plotting.
plt.figure(figsize=(32,32))
plt.subplot(1,2,1)
plt.imshow(ch0)
#plt.subplot(1,3,2)
#plt.imshow(ch1)
plt.subplot(1,2,2)
plt.imshow(ch2)

#Otsu's thersholding based on DAB
thresh = threshold_otsu(ch2)
binary = ch2 < thresh


# =============================================================================
# # Adaptive Gaussian Thresholding
# th1 = cv2.adaptiveThreshold(gray_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#         cv2.THRESH_BINARY,11,2)
# 
# # Otsu's thresholding after Gaussian filtering
# blur = cv2.GaussianBlur(th2,(5,5),0)
# ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# 
# =============================================================================
#Save the thresholded image into an output (first transform into uint8 since pil needs this)
img = Image.fromarray((binary * 255).astype(np.uint8))
plt.imshow(img)

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
# # show it
# plt.imshow(binary, cmap="gray")
# plt.show()
# 
# 
# # find the contours from the thresholded image
# contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# # draw all contours
# im = cv2.drawContours(im, contours, -1, (0, 255, 0), 2)
# 
# # show the image with the drawn contours
# plt.imshow(im)
# plt.show()
# 
# =============================================================================


# =============================================================================
# # Show user what we found
# for cnt in contours:
#    (x,y),radius = cv2.minEnclosingCircle(cnt)
#    center = (int(x),int(y))
#    radius = int(radius)
#    print('Contour: centre {},{}, radius {}'.format(x,y,radius))
#    
# =============================================================================

# =============================================================================
# #Extract the controur values and apply
# x, y = [], []
# 
# for contour_line in contours:
#     for contour in contour_line:
#         x.append(contour[0][0])
#         y.append(contour[0][1])
# 
# x1, x2, y1, y2 = min(x), max(x), min(y), max(y)
# 
# cropped = ch0[y1:y2, x1:x2]
# plt.imshow(ch0)
# 
# plt.imshow(cropped)
# 
# cnt = contours[0]
# 
# x,y,w,h = cv2.boundingRect(cnt) #good
# 
# ###apply coordinates to the hematoxylin channel
# ch0 = ch0.crop((x, y, w, h))
# =============================================================================


#approach 2
ROI = np.zeros(im.shape, np.uint8)  # Create an empty numpy array of the same size as the original image for storageROIInformation

ret, binary = cv2.threshold(gray,
    0, 255, 
    cv2.THRESH_BINARY_INV | cv2.THRESH_TRIANGLE)  # Adaptive Binarization
canny = cv2.Canny(binary, 30, 30)
cnts, _ = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

out_binary, contours, hierarchy = cv2.findContours(binary, 
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)  # Find all contours, each contour information is stored in the contours array
out_binary, contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # fails with error "not enough values to unpack (expected 3, got 2)"
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for cnt in range(len(contours)):  # Process each contour based on the number of contours
         # Contour approximation, the specific principles need to be studied in depth
    epsilon = 0.01 * cv.arcLength(contours[cnt], True)
    approx = cv.approxPolyDP(contours[cnt], epsilon, True)  # Save the vertex information of the approximation result
    							        # The number of vertices determines the contour shape 
         # Calculate the contour center position							   
    mm = cv.moments(contours[cnt])
    if mm['m00'] != 0:
        cx = int(mm['m10'] / mm['m00'])
        cy = int(mm['m01'] / mm['m00'])
        color = src[cy][cx]
        color_str = "(" + str(color[0]) + ", " + str(color[1]) + ", " + str(color[2]) + ")"
        p = cv.arcLength(contours[cnt], True)
        area = cv.contourArea(contours[cnt])
        
                 # Analyze geometry
        corners = len(approx)
        if corners == 3 and (color[2]>=150 or color[0]>=150) and area>1000:    # A series of judgment conditions are adjusted by the characteristics of the project
            cv.drawContours(ROI, contours, cnt, (255, 255, 255), -1)    # AtROIDraw an outline on the empty canvas and fill it with white (the last parameter is the width of the outline line, if it is a negative number, it will directly fill the area)
            imgroi = ROI & src  # ROIAND the original image to filter out the original imageROIarea
            cv.imshow("ROI", imgroi)
            cv.imwrite(r"D:\ROI.jpg")
            
        if corners >= 10 and (color[2]>=150 or color[0]>=150) and area>1000:          
    	    cv.drawContours(ROI, contours, cnt, (255, 255, 255), -1)
            imgroi = ROI & src
            cv.imshow("ROI",imgroi）
            cv.imwrite(r"D:\ROI.jpg")

            
cv.waitKey(0)
cv.destroyAllWindows()         


