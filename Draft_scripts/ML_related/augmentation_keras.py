#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 11:27:34 2021

@author: atte
"""
import tensorflow as tf
import keras
from tensorflow.keras.models import *
from keras.preprocessing.image import ImageDataGenerator
import cv2


#1
# Horizontal shift image augmentation
# =============================================================================
# We can see in the plot of the result that a range of different randomly selected
# positive and negative horizontal shifts was performed and the pixel values at 
# the edge of the image are duplicated to fill in the empty part of the image created
# by the shift.
# =============================================================================
import matplotlib
matplotlib.use('Agg')


from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import pathlib
import os

path = '/home/inf-54-2020/experimental_cop/H_final/Images/'
outp = '/home/inf-54-2020/experimental_cop/H_final/Img_and_augm/'
for file in os.listdir(path):

     if file.endswith(".jpg"): 
        imagepath=path + "/" + file
        img = load_img(imagepath)
        #imgname = ntpath.basename(imagepath)#
        
        # convert to numpy array
        data = img_to_array(img)
        # expand dimension to one sample
        samples = expand_dims(data, 0)
        # create image data augmentation generator
        datagen = ImageDataGenerator(width_shift_range=[-200,200])
        # prepare iterator
        it = datagen.flow(samples, batch_size=1)
        # generate samples and plot
        for i in range(9):
            # define subplot
            #pyplot.subplot(330 + 1 + i)
            pyplot.axis('off')
        
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            # plot raw pixel data
            pyplot.imshow(image)
            #image.save(path + '{}_Training20xgraft.tif.format(i)') 
            pyplot.savefig(outp + str(i) + '_' + 'hor_shift' + '_' + file, bbox_inches='tight')
            pyplot.show()
        
        
        #2
        # Vertical shift image augmentation
        # =============================================================================
        # creates a plot of images augmented with random positive and negative vertical shifts
        # =============================================================================
        
        # create image data augmentation generator
        datagen = ImageDataGenerator(height_shift_range=0.5)
        # prepare iterator
        it = datagen.flow(samples, batch_size=1)
        # generate samples and plot
        for i in range(9):
            # define subplot
            #pyplot.subplot(330 + 1 + i)
            pyplot.axis('off')
        
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            # plot raw pixel data
            pyplot.imshow(image)
            #image.save(imagepath + '{}_Training20xgraft.tif.format(i)') 
            pyplot.savefig(outp + str(i) + '_' 'ver_shift' + '_' + file, bbox_inches='tight')
            pyplot.show()
        
        #3
        # Vertical flip augmentation
        datagen = ImageDataGenerator(horizontal_flip=True)
        # prepare iterator
        it = datagen.flow(samples, batch_size=1)
        # generate samples and plot
        for i in range(9):
            # define subplot
            #pyplot.subplot(330 + 1 + i)
            pyplot.axis('off')
        
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            # plot raw pixel data
            pyplot.imshow(image)
            #image.save(imagepath + '{}_Training20xgraft.tif.format(i)') 
            pyplot.savefig(outp + str(i) +'_' +'ver_flip' + '_' + file, bbox_inches='tight')
        
            pyplot.show()
        #4
        # Horizontal flip augmentation
        datagen = ImageDataGenerator(vertical_flip=True)
        # prepare iterator
        it = datagen.flow(samples, batch_size=1)
        # generate samples and plot
        for i in range(9):
            # define subplot
            #pyplot.subplot(330 + 1 + i)
            pyplot.axis('off')
        
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            # plot raw pixel data
            pyplot.imshow(image)
            #image.save(imagepath + '{}_Training20xgraft.tif.format(i)') 
            pyplot.savefig(outp + str(i) + '_' + 'hor_flip' + '_' + file, bbox_inches='tight')
            pyplot.show()
        
        #5
        # Random rotation
        # =============================================================================
        # generates rotated images, showing in some cases pixels rotated out of the frame
        # and the nearest-neighbor fill.
        # =============================================================================
        
        # create image data augmentation generator
        datagen = ImageDataGenerator(rotation_range=300)
        # prepare iterator
        it = datagen.flow(samples, batch_size=1)
        # generate samples and plot
        for i in range(9):
            # define subplot
            #pyplot.subplot(330 + 1 + i)
            pyplot.axis('off')
        
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            # plot raw pixel data
            pyplot.imshow(image)
            #image.save(imagepath + '{}_Training20xgraft.tif.format(i)') 
            pyplot.savefig(outp + str(i) +  '_' + 'rotat' + '_' + file, bbox_inches='tight')
            #pyplot.show()
        
        #6
        # Random brightness adjustments
        # =============================================================================
        # The brightness of the image can be augmented by either randomly darkening images, 
        # brightening images, or both. The intent is to allow a model to generalize across
        # images trained on different lighting levels.
        # =============================================================================
        
        # create image data augmentation generator
        datagen = ImageDataGenerator(brightness_range=[0.2,1.5]) #define the range of brigthness
        #values > 1 darken the image, <1 brighten, and just 1 is neutral
        # prepare iterator
        it = datagen.flow(samples, batch_size=1)
        # generate samples and plot
        for i in range(9):
            # define subplot
            #pyplot.subplot(330 + 1 + i)
            pyplot.axis('off')
        
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            # plot raw pixel data
            pyplot.imshow(image)
            #image.save(imagepath + '{}_Training20xgraft.tif.format(i)') 
            pyplot.savefig(outp + str(i) +  '_' + 'brightn' + '_' + file, bbox_inches='tight')
            pyplot.show()
        
        #7
        #Random zoom augmentation
        # =============================================================================
        # A zoom augmentation randomly zooms the image in and either adds new pixel values 
        # around the image or interpolates pixel values respectively.
        # # create image data augmentation generator
        # =============================================================================
        datagen = ImageDataGenerator(zoom_range=[0.5,1.0])
        # prepare iterator
        it = datagen.flow(samples, batch_size=1)
        # generate samples and plot
        for i in range(9):
            # define subplot
            #pyplot.subplot(330 + 1 + i)
            pyplot.axis('off')
        
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            # plot raw pixel data
            pyplot.imshow(image)
            #image.save(imagepath + '{}_Training20xgraft.tif.format(i)') 
            pyplot.savefig(outp + str(i) +  '_' + 'zoom' + '_' + file, bbox_inches='tight')
            pyplot.show()
         
     else:
         continue

