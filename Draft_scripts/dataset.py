#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 15:16:58 2021

@author: atte
"""

import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class Nuclei(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir) #lists all the files from the directory

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index]) #with index you get the certain image 
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".tif", "_Mask.tif"))
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) #transform the mask into np array. L makes it into grayscale (PIL feature)
        mask[mask == 255.0] = 1.0 #we look for the point in mask where its 255 and make it into 1 since
        # we use a sigmoid function as activation function. Indicates p that the pixel value is white
# =============================================================================
# 
#         if self.transform is not None: #performs data augmentation (not necessary if you've already done this)
#             augmentations = self.transform(image=image, mask=mask)
#             image = augmentations["image"]
#             mask = augmentations["mask"]
# 
# =============================================================================
        return image, mask
