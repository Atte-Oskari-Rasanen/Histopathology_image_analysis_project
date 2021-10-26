#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 17:03:45 2021

@author: atte
"""
#http://creativemorphometrics.co.vu/blog/2014/08/05/automated-outlines-with-opencv-in-python/
#https://stackoverflow.com/questions/28759253/how-to-crop-the-internal-area-of-a-contour
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.filters import threshold_otsu, threshold_local

img = cv2.imread('/home/atte/Documents/images_qupath2/batch4/deconv/test/14_3._D._BW.tif') # Read in your image

from PIL import Image
#img = Image.open('/home/atte/Documents/images_qupath2/batch4/deconv/test/14_3._D._BW.tif').convert('LA')
#img = np.asarray(img)
ret,thresh1 = cv2.threshold(img,15,255,cv2.THRESH_BINARY) #the value of 15 is chosen by trial-and-error to produce the best outline of the skull
kernel = np.ones((5,5),np.uint8) #square image kernel used for erosion
erosion = cv2.erode(thresh1, kernel,iterations = 1) #refines all edges in the binary image

opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel) #this is for further removing small noises and holes in the image

plt.imshow(closing, 'gray') #Figure 2
plt.xticks([]), plt.yticks([])
plt.show()

binary = (closing > 0).astype(int)

#convert img to grayscale. already should be but for some reason if this step is skipped,
#the contours line wont work:
closing = cv2.cvtColor(closing, cv2.COLOR_BGR2GRAY)

contours, hierarchy = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #find contours with simple approximation
plt.imshow(closing)

areas = [] #list to hold all areas

for contour in contours:
  ar = cv2.contourArea(contour)
  areas.append(ar)

max_area = max(areas)
max_area_index = areas.index(max_area) #index of the list element with largest area

cnt = contours[max_area_index] #largest area contour

cv2.drawContours(closing, [cnt], 0, (255, 255, 255), 3, maxLevel = 0)
plt.imshow(closing)




cv2.imshow('cleaner', closing)




contours, hierarchy = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #find contours with simple approximation

cv2.imshow('cleaner', closing) #Figure 3
cv2.drawContours(closing, contours, -1, (255, 255, 255), 4)
cv2.waitKey(0)


#gwashBW = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #change to grayscale


plt.imshow(gwashBW, 'gray') #this is matplotlib solution (Figure 1)
plt.xticks([]), plt.yticks([])
plt.show()

cv2.imshow('gwash', gwashBW) #this is for native openCV display
cv2.waitKey(0)


# thresh = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
# thresh = threshold_otsu(img)
# binary = img > thresh

bgr = cv2.imread('/home/atte/Documents/images_qupath2/batch4/deconv/test/14_3._H.tif') # Read in your image

bgr = cv2.imread('puzzle.png')
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
_, roi = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
cv2.imwrite('/home/dhanushka/stack/roi.png', roi)
plt.imshow(roi)
cont = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
output = np.zeros(gray.shape, dtype=np.uint8)
cv2.drawContours(output, cont[0], -1, (255, 255, 255))
plt.imshow()
# removing boundary
boundary = 255*np.ones(gray.shape, dtype=np.uint8)
boundary[1:boundary.shape[0]-1, 1:boundary.shape[1]-1] = 0

toremove = output & boundary
output = output ^ toremove


import cv2
import numpy as np

# read chessboard image
img = cv2.imread('chessboard.png')

# read pawn image template
template = cv2.imread('/home/atte/Documents/images_qupath2/batch4/deconv/test/14_3._D._BW.tif') # Read in your image

template = cv2.imread('pawn.png', cv2.IMREAD_UNCHANGED)
hh, ww = template.shape[:2]

# extract pawn base image and alpha channel and make alpha 3 channels
pawn = template[:,:,0:3]
alpha = template[:,:,3]
alpha = cv2.merge([alpha,alpha,alpha])

# do masked template matching and save correlation image
correlation = cv2.matchTemplate(img, pawn, cv2.TM_CCORR_NORMED, mask=alpha)

# set threshold and get all matches
threshhold = 0.89
loc = np.where(correlation >= threshhold)

# draw matches 
result = img.copy()
for pt in zip(*loc[::-1]):
    cv2.rectangle(result, pt, (pt[0]+ww, pt[1]+hh), (0,0,255), 1)
    print(pt)

#contours, _ = cv2.findContours(...) # Your call to find the contours using OpenCV 2.4.x
# contours, hierarchy = cv2.findContours(th2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

ret, binary = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

plt.imshow(contours)

# Find contour and sort by contour area
cnts = cv2.findContours(template, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

# Find bounding box and extract ROI
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    ROI = D_img[y:y+h, x:x+w]
    break

imgplot = plt.imshow(ROI)
plt.show()
#draw all contours
d = cv2.drawContours(D_img, contours, -1, (0,255,0), 3)
imgplot = plt.imshow(d)

#To draw an individual contour, say 4th contour:
cv2.drawContours(img, contours, 3, (0,255,0), 3)
imgplot = plt.imshow(img)

#But most of the time, below method will be useful:
cnt = contours[4]
cv2.drawContours(img, [cnt], 0, (0,255,0), 3)

cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 600,600)

cv2.imshow('w',contours)
cv2.imshow('original', img)
cv2.waitKey() 
cv2.destroyAllWindows()
_, contours, _ = cv2.findContours(th2) # Your call to find the contours
idx = 0 # The index of the contour that surrounds your object
mask = np.zeros_like(img) # Create mask where white is what we want, black otherwise
cv2.drawContours(mask, contours, idx, 255, -1) # Draw filled contour in mask
plt.imshow(mask)
out = np.zeros_like(img) # Extract out the object and place into output image
H_img = cv2.imread('/home/atte/Documents/images_qupath2/batch4/deconv/test/14_3._H.tif', 0) # Read in your image

out[mask == 255] = H_img[mask == 255]

# Show the output image
cv2.imshow('Output', out)
cv2.waitKey(0)
cv2.destroyAllWindows()
