#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 13:19:38 2021

@author: atte
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

from skimage import data
from skimage.color import rgb2hed, hed2rgb



bg = cv2.imread("/home/atte/Pictures/Deconv_im/20xnorm")
graft = cv2.imread("/home/atte/Pictures/Deconv_im/20xgraft")
bg.show() #cant see anything since it's directly transformed into a numpy array


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



# Example IHC image
ihc_rgb = cv2.imread('./images_qupath2/YZ004_NR_G2_#15_hCOL1A1_20x_(2).tif')

# Separate the stains from the IHC image
ihc_hed = rgb2hed(ihc_rgb)

# Create an RGB image for each of the stains
null = np.zeros_like(ihc_hed[:, :, 0])
ihc_h = hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1))
ihc_e = hed2rgb(np.stack((null, ihc_hed[:, :, 1], null), axis=-1))
ihc_d = hed2rgb(np.stack((null, null, ihc_hed[:, :, 2]), axis=-1))

# Display
fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(ihc_rgb)
ax[0].set_title("Original image")

ax[1].imshow(ihc_h)
ax[1].set_title("Hematoxylin")

ax[2].imshow(ihc_e)
ax[2].set_title("Eosin")  

ax[3].imshow(ihc_d)
ax[3].set_title("DAB")

for a in ax.ravel():
    a.axis('off')

fig.tight_layout()


from skimage.exposure import rescale_intensity

# Rescale hematoxylin and DAB channels and give them a fluorescence look
h = rescale_intensity(ihc_hed[:, :, 0], out_range=(0, 1), #blue - hematoxylin
                      in_range=(0, np.percentile(ihc_hed[:, :, 0], 99)))
d = rescale_intensity(ihc_hed[:, :, 2], out_range=(0, 1),
                      in_range=(0, np.percentile(ihc_hed[:, :, 2], 99))) #red - dab

# Cast the two channels into an RGB image, as the blue and green channels
# respectively
zdh = np.dstack((d, null, d))
#no proper separation due to the overlap


fig = plt.figure()
axis = plt.subplot(1, 1, 1, sharex=ax[0], sharey=ax[0])
axis.imshow(zdh)
axis.set_title('Stain-separated image (blue-h, red - d)')
axis.axis('off')
plt.show()
