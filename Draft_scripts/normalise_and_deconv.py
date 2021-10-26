#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 13:37:50 2021

@author: atte
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('/home/atte/Pictures/HE_DAB1.png')

#opencv reads images as BGR but we need to convert to RGB since its important with cell images
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


Io=240 #transmitted light intensity, normalising factor for image intensity
alpha= 1 #tolerance for the pseudo-min and pseudo-max (default: 1)
beta=0.15  #the threshold value for pixels to remove

#0.527 0.606 0.596 

#standard values taken from a paper, better ones can be applied if available
HEref = np.array([[0.52626, 0.2159], [0.7201, 0.8012], [0,4062, 0.5581]])

maxCref = np.array([1.9705,1.0308])


#extract the height, width and num of channels of image
h,w,c = img.shape

#reshape img to multiple rows and 3 columns
#the row number depends on the image size
img = img.reshape(-1,3)

#Optical density
OD = np.log10((img.astype(np.float)+1)/Io)
#had to add 1 to all floats since if the pixel value is 0 at some point,
#the log cant be calculated


#if you import imgs with skimage instead:
#OD = -log10(I)
#OD = -np.log10(img+0.004)

'''
#3d plot
#####
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(16,8))
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(img[:,0],img[:,1],img[:,2])
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(OD[:,0],OD[:,1],OD[:,2])
plt.show()
#####
'''
#2
#remove data with OD intensity < 6  ---- removed all!
ODhat = OD[~np.any(OD < beta, axis = 1)]

#3 
#Calculate SVD on the OD tuples
#estimate covar matrix of ODhat
eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))


That = ODhat.dot(eigvecs[:,1:3])

phi = np.arctan2(That[:,1], That[:,0])

minPhi = np.percentile(phi, alpha)
maxPhi = np.percentile(phi, 100-alpha)

vMin = eigvecs[:,1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
vMax = eigvecs[:,1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)

if vMin[0] > vMax[0]:
    HE = np.array((vMin[:,0], vMax[:,0]).T)
else:
    HE = np.array((vMax[:,0], vMin[:,0]).T)

#Rows = channels(RBG), columns = OD values 
Y = np.reshape(OD, (-1, 3)).T

#find out the c of the individual stains
C = np.linalg.lstsq(HE,Y, rcond=None)[0]

maxC=np.array([np.percentile(C[0,:],99), np.percentile(C[1,:]), 99])

tmp = np.divide(maxC,maxCref)
C2 = np.divide(C,tmp[:, np.newaxis])

###### Step 8: Convert extreme values back to OD space
# recreate the normalized image using reference mixing matrix 

Inorm = np.multiply(Io, np.exp(-HEref.dot(C2)))
Inorm[Inorm>255] = 254
Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)  

# Separating H and E components

H = np.multiply(Io, np.exp(np.expand_dims(-HEref[:,0], axis=1).dot(np.expand_dims(C2[0,:], axis=0))))
H[H>255] = 254
H = np.reshape(H.T, (h, w, 3)).astype(np.uint8)

E = np.multiply(Io, np.exp(np.expand_dims(-HEref[:,1], axis=1).dot(np.expand_dims(C2[1,:], axis=0))))
E[E>255] = 254
E = np.reshape(E.T, (h, w, 3)).astype(np.uint8)

plt.imsave("./Pictures/Deconv_im/HnE_normalized.jpg", Inorm)
plt.imsave("./Pictures/Deconv_im/HnE_separated_H.jpg", H)
plt.imsave("./Pictures/Deconv_im/HnE_separated_E.jpg", E)






