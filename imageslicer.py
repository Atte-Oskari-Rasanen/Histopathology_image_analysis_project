#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 13:37:57 2021

@author: atte
"""

from PIL import Image
from numpy import np
import image_slicer

large_image = Image.open('/home/atte/Documents/googletest.jpeg')
large_image=np.array(large_image)
h = large_image.shape[1]
new_size = (h,h,3)
im = np.resize(large_image,new_size)
image_slicer.slice(im, 14)



tiles = [im[x:x+M,y:y+N] for x in range(0,im.shape[0],M) for y in range(0,im.shape[1],N)]

img = crop(large_image, h, h, 0, )
print(large_image.shape)
