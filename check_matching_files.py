#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 11:15:13 2021

@author: atte
"""
import os
TRAIN_IMG_DIR = "/home/inf-54-2020/experimental_cop/Train_H_Final/Full_Aug_Img/"
M_TRAIN_IMG_DIR = "/home/inf-54-2020/experimental_cop/Train_H_Final/Full_Aug_Mask/"

Imgs = []
Masks = []
for imagefile in os.listdir(TRAIN_IMG_DIR):  #to go through files in the specific directory
    ending = imagefile.split('_')[2]
    Imgs.append(ending)
for imagefile in os.listdir(M_TRAIN_IMG_DIR):  #to go through files in the specific directory
    ending = imagefile.split('_')[2]
    Masks.append(ending)

not_found = []
not_found2 =[]
for m in Masks:
    if m in Imgs:
        continue
    else:
        not_found.append(m)

for i in Imgs:
    if i in Masks:
        continue
    else:
        not_found2.append(i)

print('Lacking pairs (masks):')
print(not_found)
print('Lacking pairs (images):')
print(not_found2)