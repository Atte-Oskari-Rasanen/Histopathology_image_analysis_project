#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 14:36:11 2021

@author: atte
"""
import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.color import rgb2hed, hed2rgb
import matplotlib.pyplot as plt

from scipy.signal import convolve2d as conv2

from skimage import color, data, restoration

from PIL import Image
import cv2

import staintools as staintools
#ihc_rgb = Image.open("./Pictures/cells.tif");

# Read data - takes in the graft area and the area against which it will be normalized 
bg = cv2.imread("/home/atte/Pictures/Deconv_im/20xnorm")
graft = cv2.imread("/home/atte/Pictures/Deconv_im/20xgraft")
bg.show() #cant see anything since it's directly transformed into a numpy array

from skimage import img_as_ubyte

bg = img_as_ubyte(bg)
cv2.imshow("Window", bg)


bg = data.astype(np.uint8)
graft = data.astype(np.uint8)

#a simpler normalization method
#info = np.iinfo(data.dtype) # Get the information of the incoming image type
#data = data.astype(np.float64) / info.max # normalize the data to 0 - 1
#data = 255 * data # Now scale by 255

# Standardize brightness (optional, can improve the tissue mask calculation)
bg = staintools.LuminosityStandardizer.standardize(bg)
# LuminosityStandardizer enforces an image to have at least 5% of pixels being luminosity saturated
graft = staintools.LuminosityStandardizer.standardize(graft)

# Stain normalize
normalizer = staintools.StainNormalizer(method='vahadane')
normalizer.fit(bg)
transformed = normalizer.transform(graft)


transformed = Image.fromarray(transformed)
transformed.save('./Documents/transformed.tif')

transformed.show();

# Read data
to_augment = staintools.read_image("./Documents/transformed.tif")

# Standardize brightness (This step is optional but can improve the tissue mask calculation)
to_augment = staintools.LuminosityStandardizer.standardize(to_augment)

# Stain augment
# =============================================================================
# augmentor = staintools.StainAugmentor(method='vahadane', sigma1=0.2, sigma2=0.2)
# augmentor.fit(to_augment)
# augmented_images = []
# for _ in range(100):
#     augmented_image = augmentor.pop()
#     augmented_images.append(augmented_image)
# 
# =============================================================================
import matplotlib.pyplot as plt

from scipy.signal import convolve2d as conv2

from skimage import color, data, restoration

rng = np.random.default_rng()

astro = color.rgb2gray(data.ihc_rgb())

psf = np.ones((5, 5)) / 25
ihc_rgb = conv2(ihc_rgb, psf, 'same')
# Add Noise to Image
astro_noisy = astro.copy()
astro_noisy += (rng.poisson(lam=25, size=astro.shape) - 10) / 255.

# Restore Image using Richardson-Lucy algorithm
deconvolved_RL = restoration.richardson_lucy(astro_noisy, psf, num_iter=30)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 5))
plt.gray()

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
