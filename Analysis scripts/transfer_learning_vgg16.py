#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 23:08:17 2021

@author: atte
"""
import tensorflow as tf
import os
import random
import numpy as np
import keras
# from tensorflow import keras
from tensorflow import keras 
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
# from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import vgg16

import matplotlib
matplotlib.use('Agg')

from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import sys
from tensorflow.keras.optimizers import Adam
from U_net_function import * 
from PIL import Image
from import_images_masks_patches import *

TRAIN_PATH = sys.argv[1]
MASK_PATH = sys.argv[2]

IMG_PROP = int(sys.argv[3])
IMG_HEIGHT = IMG_WIDTH = IMG_PROP



X_train = import_images(TRAIN_PATH, IMG_HEIGHT,IMG_WIDTH, 3)
Y_train = import_masks(MASK_PATH, IMG_HEIGHT,IMG_WIDTH)



X_train = X_train.astype(np.float64)
Y_train = Y_train.astype(np.float64)

#scale data
X_train = X_train / 255
Y_train = Y_train / 255
random_no = random.randint(0, len(X_train))
im1 = X_train[random_no]
im = np.array(Image.fromarray((im1).astype(np.uint8)))
im = Image.fromarray(im)
im.save("random_im_vgg.png")


# im = '/home/inf-54-2020/experimental_cop/Train_H_Final/Train_Ims_batch5/Ims/DAB_15sec_1.tif'
# img = cv2.imread(im)[:,:,:1]
# cv2.imwrite('./test_dab_15sec1.tif', img)

# print('done!')
def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

seed = 42
new_input = Input(shape=(IMG_PROP, IMG_PROP, 3))

vgg_model = tf.keras.applications.VGG16(weights='imagenet', input_tensor=new_input, include_top=False, input_shape=(IMG_PROP, IMG_PROP, 3), pooling='avg')

# Freeze four convolution blocks
for layer in vgg_model.layers[:15]:
    layer.trainable = False
# Make sure you have frozen the correct layers
for i, layer in enumerate(vgg_model.layers):
    print(i, layer.name, layer.trainable)
# so we will be training our dataset on the last four layers of the pre-trained VGG-16 model.

x = vgg_model.output
x = Flatten()(x) # Flatten dimensions to for use in FC layers
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x) # Dropout layer to reduce overfitting
x = Dense(256, activation='relu')(x)
x = Dense(8, activation='sigmoid')(x) # Softmax for multiclass
transfer_model = Model(inputs=vgg_model.inputs, outputs=x)

# transfer_model.compile(optimizer=Adam(learning_rate = 0.001), loss=[dice_coef_loss], 
#               #BinaryFocalLoss(gamma=2)
#               metrics=[dice_coef])
all_dat = int(X_train.shape[0])
print(all_dat)

import math
batch_size=1
trainingsize = round(all_dat * 0.7)
print(trainingsize)
validate_size = all_dat * 0.3
def calculate_spe(y):
  return int(math.ceil((1. * y) / batch_size))
steps_per_epoch = calculate_spe(trainingsize)
validation_steps = calculate_spe(validate_size)

cp_save_path = './transfer_learning_vgg16_ep3_retry.h5'
checkpointer = tf.keras.callbacks.ModelCheckpoint(cp_save_path, verbose=1, save_best_only=True)
tb_cb = tf.keras.callbacks.TensorBoard(log_dir='logs', profile_batch=0)
transfer_model.compile(optimizer=Adam(learning_rate = 0.0001), loss=[dice_coef_loss], 
              #BinaryFocalLoss(gamma=2)
              metrics=[dice_coef])

callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='logs')]
save_path = '/home/inf-54-2020/experimental_cop/scripts/Plots_Unet/transfer_learning_vgg16_3ep_s244.h5'
history = transfer_model.fit(X_train, Y_train, validation_split=0.5, batch_size=batch_size, epochs=1, callbacks=callbacks)
transfer_model.save(save_path)

plt.plot(history.history['dice_coef'])
plt.plot(history.history['val_dice_coef'])
plt.title('model accuracy dice')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')

figname = save_path + 'VGG16_AccuracyDice_Full_model_s244_bs1_ep3.png'
plt.show()
plt.savefig(figname)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss dice')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
figname = save_path + 'VGG16_LossDice_Full_model_s512_bs1_ep3.png'
plt.savefig(figname)

print('done!')
#Finetuning by unfreezing the fifth deconv block:
# for layer in vgg_model.layers[:15]:
#     layer.trainable = False
#     x = vgg_model.output
#     x = Flatten()(x) # Flatten dimensions to for use in FC layers
#     x = Dense(512, activation='relu')(x)
#     x = Dropout(0.5)(x) # Dropout layer to reduce overfitting
#     x = Dense(256, activation='relu')(x)
#     x = Dense(8, activation='sigmoid')(x) # Softmax for multiclass, sigmoid for binary class
# transfer_model = Model(inputs=vgg_model.input, outputs=x)
# for i, layer in enumerate(transfer_model.layers):
#     print(i, layer.name, layer.trainable)


# #Augment images
# # train_datagen = ImageDataGenerator(zoom_range=0.2, rotation_range=30, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2)
# # #Fit augmentation to training images
# # train_generator = train_datagen.flow(X_train,Y_train,batch_size=1)
# #Compile model
# transfer_model.compile(optimizer=Adam(learning_rate = 0.0001), loss=[dice_coef_loss], 
#               #BinaryFocalLoss(gamma=2)
#               metrics=[dice_coef])

# cp_save_path = './transfer_learning_vgg16_finetuned_1ep.h5'

# #Fit model
# history = transfer_model.fit(X_train, Y_train, validation_split=0.5, batch_size=1, epochs=1, callbacks=callbacks)
# transfer_model.save(cp_save_path)


# plt.plot(history.history['dice_coef'])
# plt.plot(history.history['val_dice_coef'])
# plt.title('model accuracy dice')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')

# figname = save_path + 'Finetuned_AccuracyDice_Full_model_s512_bs1_ep1.png'
# plt.show()
# plt.savefig(figname)

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss dice')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()
# figname = save_path + 'Finetuned_LossDice_Full_model_s512_bs1_ep1.png'
# plt.savefig(figname)