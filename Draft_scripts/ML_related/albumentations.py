#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 11:57:36 2021

@author: atte
"""

import random

import cv2
from matplotlib import pyplot as plt
import numpy as np
import albumentations as A
import test_augment_bboxes

#define augmentation pipeline
transform = A.Compose([
    A.RandomCrop(width=450, height=450),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
], bbox_params=A.BboxParams(format='pascal_voc'))

image = cv2.imread("/home/atte/Documents/images_qupath2/Output/H_final/YZ004 NR G2 #5 hCOL1A1 10x (1)_H_Final.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
from collections import Counter

bboxes_a = np.loadtxt("/home/atte/Documents/images_qupath2/Output/H_final/10x_1_H_Final.txt", dtype=str)
x = np.delete(bboxes_a, np.arange(0, bboxes_a.size, 5))
i = len(x)/4
bboxes = np.split(x, i)
bboxes




transformed = transform(image=image, bboxes=bboxes)
transformed_image = transformed['image']
transformed_bboxes = transformed['bboxes']