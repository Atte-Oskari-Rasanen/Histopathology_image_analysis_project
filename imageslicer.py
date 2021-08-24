#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 13:37:57 2021

@author: atte
"""

from PIL import Image
import numpy as np
import image_slicer
from skimage.io import imread, imshow

import os

def crop(infile,height,width):
    im = Image.open(infile)
    imgwidth, imgheight = im.size
    for i in range(imgheight//height):
        for j in range(imgwidth//width):
            box = (j*width, i*height, (j+1)*width, (i+1)*height)
            yield im.crop(box)

#if __name__=='__main__':
infile='/home/atte/Documents/googletest.jpeg'
# print(infile.shape)

height=500
width=500
start_num=1
print('starting the loop...')
for k,piece in enumerate(crop(infile,height,width),start_num):
    img=Image.new('RGB', (height,width), 255)
    img.paste(piece)
    path=os.path.join("IMG-%s.png" % k)
    img.save(path)
    print("image saved into: " + path)

large_image = Image.open('/home/atte/Documents/googletest.jpeg')
large_image=np.array(large_image)
h = large_image.shape[1]
new_size = (h,h,3)
im = np.resize(large_image,new_size)
image_slicer.slice(im, 14)



#####
#apply this to large images, train the model with these smaller patches
#when predicting on large images, break the image into smaller patches like this,
#then apply the processes like model.predict on these arrays, append into segm_images
#and then save as a whole slide image

import cv2
path_to_img = '/home/atte/Documents/googletest.jpeg'
save_path = '/home/atte/Documents/'
img = cv2.imread(path_to_img)
print(type(img))
img = np.asarray(img)
img_h, img_w, _ = img.shape
#img = np.resize(img, (500,500))

split_width = 50
split_height = 50


def start_points(size, split_size, overlap=0):
    points = [0]
    stride = int(split_size * (1-overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            points.append(size - split_size)
            break
        else:
            points.append(pt)
        counter += 1
    return points


X_points = start_points(img_w, split_width, 0.1)
Y_points = start_points(img_h, split_height, 0.1)

splitted_images = []

for i in Y_points:
    for j in X_points:
        split = img[i:i+split_height, j:j+split_width]
        im = Image.fromarray(split)
        im.save(save_path + str(i) + str(j) +'_remade.png')

        splitted_images.append(split)


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
