#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 13:58:55 2021

@author: atte
"""


import matplotlib.pyplot as plt
import skimage
from skimage import io
from skimage.filters.rank import entropy
from skimage.morphology import disk
import numpy as np
from skimage.filters import threshold_otsu

path = '/home/atte/Documents/images_qupath2/H_final/patches/patchesH_Final__010.png'
from skimage import io

img = io.imread(path, as_gray=True)

entropy_img = entropy(img, disk(3))
plt.imshow(entropy_img)

th = threshold_otsu(entropy_img)
binary = entropy_img <= th
plt.imshow(binary)
plt.imshow(img)
