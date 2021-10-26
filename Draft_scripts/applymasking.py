#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 12:01:49 2021

@author: atte
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_otsu

# Load images as grayscale
H = cv2.imread('/home/atte/Pictures/H_test.png', cv2.IMREAD_GRAYSCALE)
D = cv2.imread('/home/atte/Pictures/D_test.png', cv2.IMREAD_GRAYSCALE)

# Create a mask using the DAPI image and binary thresholding at 25
D = np.array(D)
thresh = threshold_otsu(D)
mask = D > thresh

# Calculate the histogram using the H image and the obtained binary mask
#hist = cv2.calcHist([H], [0], mask, [256], [0, 256])

# Show bar plot of calculated histogram
#plt.bar(np.arange(256), np.squeeze(hist))
#plt.show()

# Show mask image
cv2.imshow('Mask', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
