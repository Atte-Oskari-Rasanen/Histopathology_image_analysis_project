#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 13:37:57 2021

@author: atte
"""

from PIL import Image
import numpy as np

#####
#apply this to large images, train the model with these smaller patches
#when predicting on large images, break the image into smaller patches like this,
#then apply the processes like model.predict on these arrays, append into segm_images
#and then save as a whole slide image

import cv2
path_to_img = '/home/atte/Documents/googletest.jpeg'
path_to_img = '/home/atte/Documents/images_qupath2/H_final/YZ004_NR_G2_#15_hCOL1A1_10x__1_H_Final.jpg'
save_path = '/home/atte/Documents/'
img = cv2.imread(path_to_img)
print(type(img))
img = np.asarray(img)
img_h, img_w, _ = img.shape
#img = np.resize(img, (500,500))

split_width = 128
split_height = 128


def start_points(size, split_size, overlap=0):
    #print(size)
    points = [0]
    stride = int(split_size * (1-overlap)) #stride equals to the split_size times overlap
    counter = 1
    while True:
        pt = stride * counter #get the multiple of the stride, i.e. the patch at hand
        #print(pt)
        if pt + split_size >= size: #if the step atm + split_size is greater than the overall size
            points.append(size - split_size)
            print(size-split_size)
            break
        else:
            print(pt)
            points.append(pt)
        counter += 1
    return points


X_points = start_points(img_w, split_width, 0.1)
print('========')
Y_points = start_points(img_h, split_height, 0.1)

splitted_images = []

for i in Y_points:
    print(i)
    print('======')
    for j in X_points:
        print(j)
        print('------')
        split = img[i:i+split_height, j:j+split_width]
        im = Image.fromarray(split)
        #im.save(save_path + str(i) + str(j) +'_remade.png')

        splitted_images.append(split)
segm_patches = []
for patch in splitted_images:
    print(patch)
    #segm = model_segm.predict(patch)
    #print(segm.shape)
    #segm_patches.append(segm)


#rebuild phase
import numpy as np
final_image = np.zeros_like(img)

index = 0
for i in Y_points:
    for j in X_points:
        final_image[i:i+split_height, j:j+split_width] = splitted_images[index]
        index += 1

im = Image.fromarray(final_image)
im.save(save_path + '/remade.png')

#cv2.imwrite(save_path, final_image)
45
90
135
180
225
270
315
360
405
450
495
540
585
630
675
720
765
810
855
900
945

#####
45
90
135
180
225
270
315
360
405
450
495
540
585
630
675