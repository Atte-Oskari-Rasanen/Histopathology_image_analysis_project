#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 16:06:48 2021

@author: atte
"""

import cv2
import numpy as np
path_to_img = '/home/atte/Desktop/outputs/Segmented/'

# Open a typical 24 bit color image. For this kind of image there are
# 8 bits (0 to 255) per color channel
img = cv2.imread(path_to_img + 'S_Hu_D_20min_10x.png')  # mandrill reference image from USC SIPI
w= img.shape[0]
h=img.shape[1]

#s = 128
#img = cv2.resize(img, (s,s), 0, 0, cv2.INTER_AREA)

def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
    
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


font = cv2.FONT_HERSHEY_SIMPLEX
fcolor = (0,0,0)

blist = [0, -127, 127,   0,  0, 64] # list of brightness values
clist = [0,    0,   0, -64, 64, 64] # list of contrast values

out = np.zeros((w*2, h*3, 3), dtype = np.uint8)

for i, b in enumerate(blist):
    c = clist[i]
    print('b, c:  ', b,', ',c)
    row = w*int(i/3)
    col = h*(i%3)
    
    print('row, col:   ', row, ', ', col)
    
    # out[row:row+s, col:col+s] = apply_brightness_contrast(img, b, c)
    # msg = 'b %d' % b
    # cv2.putText(out,msg,(col,row+s-22), font, .7, fcolor,1,cv2.LINE_AA)
    # msg = 'c %d' % c
    # cv2.putText(out,msg,(col,row+s-4), font, .7, fcolor,1,cv2.LINE_AA)
    
    # cv2.putText(out, 'OpenCV',(260,30), font, 1.0, fcolor,2,cv2.LINE_AA)
b=64
c=64
output = out[row:row+w, col:col+h] = apply_brightness_contrast(img, b, c)
cv2.imwrite(path_to_img + 'out_bc64.png', output)
