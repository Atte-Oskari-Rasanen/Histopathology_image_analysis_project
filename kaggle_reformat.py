#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 20:32:35 2021

@author: atte
"""

from PIL import Image
import numpy as np
np_path = '/home/inf-54-2020/experimental_cop/scripts/np_data/'
X_train_k = np.load(np_path +'X_Dataset_k_s128.npy')
Y_train_k = np.load(np_path +'Y_Dataset_k_s128.npy')

kaggle_images = '/home/inf-54-2020/experimental_cop/Train_H_Final/Train_by_batches/'
kaggle_masks = '/home/inf-54-2020/experimental_cop/Train_H_Final/Masks_by_batches/'

ki = 'kaggle_img_'
km = 'kaggle_mask_'
k = 0
m = 0
print(X_train_k.shape)
for im in X_train_k:
    print(im.shape)
    #im = np.squeeze(im, axis=2)
    img = Image.fromarray((im * 255).astype(np.uint8))
    fname = kaggle_images + ki + str(k) + '.png'
    img.save(fname)
    k += 1
    print(k)
for im in Y_train_k:
    print(im.shape)
    im = np.squeeze(im, axis=2)
    img = Image.fromarray((im * 255).astype(np.uint8))
    fname = kaggle_masks + km + str(m) + '.png'
    img.save(fname)
    m += 1
    print(m)

print('done!')

