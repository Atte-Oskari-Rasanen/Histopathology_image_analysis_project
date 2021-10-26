#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 14:33:13 2021

@author: atte
"""

import tensorflow as tf
import os
import random
import numpy as np
import keras
# from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import schedules
from tensorflow.keras import Model
import matplotlib
matplotlib.use('Agg')

from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import sys
from tensorflow.keras.optimizers import Adam
from import_images_masks_patches import *
from U_net_function import * 
from PIL import Image
for device in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(device, True)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


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
np.random.seed = seed
def normalize(img):
    min = img.min()
    max = img.max()
    x = 2.0 * (img - min) / (max - min) - 1.0
    return x
def save_random_im(X_train, Y_train):
    random_no = random.randint(0, len(X_train))
    im1 = X_train[random_no]
    im = np.array(Image.fromarray((im1 * 255).astype(np.uint8)))
    im = Image.fromarray(im)
    im.save("random_im.png")
    
    mask1 = Y_train[random_no]
    mask = np.array(Image.fromarray((mask1 * 255).astype(np.uint8)))
    mask = Image.fromarray(mask)
    mask.save("random_mask.png")

TRAIN_PATH = sys.argv[1]
MASK_PATH = sys.argv[2]

IMG_PROP = int(sys.argv[3])
IMG_HEIGHT = IMG_WIDTH = IMG_PROP
# IMG_HEIGHT = int(sys.argv[3])
# IMG_WIDTH = int(sys.argv[4])

IMG_CHANNELS = 3

# TRAIN_PATH = '/home/inf-54-2020/experimental_cop/Train_H_Final/Train_by_batches/Images/'
# MASK_PATH = '/home/inf-54-2020/experimental_cop/Train_H_Final/Masks_by_batches/Masks/'

X_train = import_images(TRAIN_PATH, IMG_HEIGHT,IMG_WIDTH, 3)
Y_train = import_masks(MASK_PATH, IMG_HEIGHT,IMG_WIDTH)

print("dtype X:", X_train.dtype)
print("dtype Y:", Y_train.dtype)

# def Datascaling(data):
#     info = np.iinfo(data.dtype) # Get the information of the incoming image type
#     print(info)
#     data = data.astype(np.float64) / info.max # normalize the data to 0 - 1
#     data = 255 * data # Now scale by 255
#     print(data[:10])
#     #scaled_data = data.astype(np.uint8)
#     return(data)

# X_train = Datascaling(X_train)
# Y_train = Datascaling(Y_train)

#need to transform to float64, otherwise gives an error when fitting the model!
# print(X_train[:10])
# print(Y_train[:10])


X_train = X_train.astype(np.float64)
Y_train = Y_train.astype(np.float64)

print(X_train[:5])
print(Y_train[:5])

# print(X_train.dtype)
# print(Y_train.dtype)

# print(type(X_train))
# print(type(Y_train))
#Y_train = np.expand_dims(Y_train, 3)

#need to transform to float32, otherwise gives an error!
# X_train = np.array(X_train, dtype=np.float32)
# Y_train = np.array(list(Y_train[:, 1]), dtype=np.float32)


# X_train = X_train.astype(np.uint8)*255
# Y_train = Y_train.astype(np.uint8)*255


# random_no = random.randint(0, len(X_train))
# im1 = X_train[random_no]
# im = np.array(Image.fromarray((im1).astype(np.uint8)))
# im = Image.fromarray(im)
# im.save("random_im.png")


print(X_train.shape)
print(Y_train.shape)

# train_ids = next(os.walk(TRAIN_PATH))[1]

# X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
# Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

# print('Resizing training images and masks')
# for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):   
#     path = TRAIN_PATH + id_
#     img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]  
#     img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
#     X_train[n] = img  #Fill empty X_train with values from img
#     mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
#     for mask_file in next(os.walk(path + '/masks/'))[2]:
#         mask_ = imread(path + '/masks/' + mask_file)
#         mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',  
#                                       preserve_range=True), axis=-1)
#         mask = np.maximum(mask, mask_)  
            
#     Y_train[n] = mask   

# print('Done!')




#Build the model
inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

#Contraction path
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
 
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
 
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
 
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

#Expansive path 
u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
 
u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
 
u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
 
u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
 
outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
 
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
from focal_loss import BinaryFocalLoss
#model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer=Adam(learning_rate = 0.00001), loss=[dice_coef_loss], 
              #BinaryFocalLoss(gamma=2)
              metrics=[dice_coef])

model.summary()

cp_save_path1 = './Full_Model_5ep_standard_metrics.h5'
cp_save_path2 = './Full_Model_10ep_standard_metrics.h5'
cp_save_path3 = './Full_Model_25ep_standard_metrics.h5'

cp_save_path1 = './Full_Model_5ep_dice_focal_s128.h5'
cp_save_path2 = './Full_Model_10ep_dice_focal_s128.h5'
cp_save_path3 = './Full_Model_25ep_dice_focal_s128.h5'

def plot_training(save_path, history, n, IMG_PROP):
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']
    # epochs = range(1, len(loss) + 1)
    # plt.plot(epochs, loss, 'y', label='Training loss')
    # plt.plot(epochs, val_loss, 'r', label='Validation loss')
    # plt.title('Training and validation loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
# summarize history for accuracy

    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    
    # figname = save_path + 'Dice_Accuracy_' + str(n) + '_ps_' + str(IMG_PROP) + '_1_Full_model_s512.png'
    
    plt.plot(history.history['dice_coef'])
    plt.plot(history.history['val_dice_coef'])
    plt.title('model accuracy dice')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')

    figname = save_path + 'AccuracyDice_' + str(n) + '_ps_' + str(IMG_PROP) + '_1_owndat_model_s512.png'
    # plt.show()
    # plt.savefig(figname)
    # # summarize history for loss
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # # plt.show()
    # figname = save_path + 'DBF_Loss_' + str(n) + '_ps_' + str(IMG_PROP) +  '_2_Full_model_s512.png'
    plt.show()

    plt.savefig(figname)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss dice')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train_Accuracy', 'Validation_Accuracy', 'Train_Loss', 'Validation_Los'], loc='upper left')
    plt.show()

    figname = save_path + 'DiceLoss_' + str(n) + '_ps' + str(IMG_PROP) + '_1_owndat_ForPlots.png'
    plt.savefig(figname)


    # jc = history.history['dice_coef']
    # #acc = history.history['accuracy']
    # val_jc = history.history['val_dice_coef']
    # #val_acc = history.history['val_accuracy']
    
    # plt.plot(epochs, jc, 'y', label='Training Dice Coeff.')
    # plt.plot(epochs, val_jc, 'r', label='Validation Dice Coeff.')
    # plt.title('Training and validation Jacard')
    # plt.xlabel('Epochs')
    # plt.ylabel('Dice Coefficient')
    # plt.legend()
    

print('Done.')

################################
#Modelcheckpoint
checkpointer = tf.keras.callbacks.ModelCheckpoint(cp_save_path1, verbose=1, save_best_only=True)
tb_cb = tf.keras.callbacks.TensorBoard(log_dir='logs', profile_batch=0)

def noop():
    pass

tb_cb._enable_trace = noop

callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='logs')]


# batch_size = 
# steps_per_epoch = X_train.shape[0] // batch_size

# from sklearn.model_selection import train_test_split

# # Split the data
# X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, X_train, test_size=0.3, shuffle= True)


#train-val split of 0.7-0.3
save_path = '/home/inf-54-2020/experimental_cop/scripts/Plots_Unet/'
# # batch size 128
# ###############
# Patchsize= "%s/augmented_image_%s.png" %(img_augmented_path, i)
# os.mkdir(n_path)  #create a new dir with the file name 
# p=save_path.rsplit('/',2)[0]
# print(p)

# history_bs128_ep5 = model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), batch_size=128, epochs=5, callbacks=callbacks)
# cp_save_path = './Plots_Unet/Full_Model_5ep_dice_diceloss_ps' + str(IMG_PROP) + '_bs128_ep5.h5'
# model.save(cp_save_path)

# history_bs128_ep10 = model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), batch_size=128, epochs=10, callbacks=callbacks)
# cp_save_path = './Plots_Unet/Full_Model_5ep_dice_diceloss_ps' + str(IMG_PROP) + '_bs128_ep10.h5'
# model.save(cp_save_path)

all_dat = int(X_train.shape[0])
print(all_dat)

import math
batch_size=128
trainingsize = round(all_dat * 0.7)
print(trainingsize)
validate_size = all_dat * 0.3
def calculate_spe(y):
  return int(math.ceil((1. * y) / batch_size))
steps_per_epoch = calculate_spe(trainingsize)
validation_steps = calculate_spe(validate_size)


#earlier run with BF loss, test the models: cp_save_path = './Plots_Unet/Owndat_dice_BFloss_ps' + str(IMG_PROP) + '_bs128_ep3.h5'

# history_bs128_ep3 = model.fit(X_train, Y_train, validation_split=0.3, batch_size=batch_size, epochs=3, callbacks=callbacks)
# cp_save_path = '/home/inf-54-2020/experimental_cop/scripts/Plots_Unet/Alldat_dice_ps' + str(IMG_PROP) + '_bs128_ep3.h5'
# model.save(cp_save_path)
# history_bs128_ep5 = model.fit(X_train, Y_train, validation_split=0.3, batch_size=batch_size, epochs=5, callbacks=callbacks)
# cp_save_path = '/home/inf-54-2020/experimental_cop/scripts/Plots_Unet/Alldat_dice_ps' + str(IMG_PROP) + '_bs128_ep5.h5'
# #cp_save_path = './Plots_Unet/Full_Model_5ep_dice_diceloss_ps' + str(IMG_PROP) + '_bs128_ep25.h5'
# model.save(cp_save_path)
history_bs128_ep10 = model.fit(X_train, Y_train, validation_split=0.3, batch_size=batch_size, epochs=10, callbacks=callbacks)
cp_save_path = '/home/inf-54-2020/experimental_cop/scripts/Plots_Unet/Alldat_10ep_dice_' + str(IMG_PROP) + '_bs128_ep10.h5'
#cp_save_path = './Plots_Unet/Full_Model_5ep_dice_diceloss_ps' + str(IMG_PROP) + '_bs128_ep25.h5'
model.save(cp_save_path)

plot_training(save_path, history_bs128_ep10, 'bs128_ep10_',str(IMG_PROP))
# history_bs128_ep50 = model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), batch_size=128, epochs=50, callbacks=callbacks)
# cp_save_path = './Plots_Unet/Full_Model_5ep_dice_diceloss_ps' + str(IMG_PROP) + '_bs128_ep50.h5'
# model.save(cp_save_path)
# plot_training(save_path, history_bs128_ep50, 'bs128_ep50', str(IMG_PROP))

# # batch size 32
# ###############
# history_bs32_ep5 = model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), batch_size=32, epochs=5, callbacks=callbacks)
# cp_save_path = './Plots_Unet/Full_Model_5ep_dice_diceloss_ps' + str(IMG_PROP) + '_bs32_ep5.h5'
# plot_training(history_bs32_ep5, 'bs32_ep5')
# model.save(cp_save_path)
# # plot_training(history1, 'ep5')
# history_bs32_ep10 = model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), batch_size=32, epochs=10, callbacks=callbacks)
# cp_save_path = './Plots_Unet/Full_Model_5ep_dice_diceloss_s512_bs32_ep10.h5'
# plot_training(history_bs32_ep10, 'bs32_ep10')
# model.save(cp_save_path)
# batch_size=32
# trainingsize = round(all_dat * 0.7)
# print(trainingsize)
# validate_size = all_dat * 0.3
# def calculate_spe(y):
#   return int(math.ceil((1. * y) / batch_size))
# steps_per_epoch = calculate_spe(trainingsize)
# validation_steps = calculate_spe(validate_size)

# history_bs32_ep25 = model.fit(X_train, Y_train, validation_split=0.3, batch_size=batch_size, epochs=100, callbacks=callbacks)
# cp_save_path = './Plots_Unet/Full_Model_5ep_dice_diceloss_s512_bs32_ep100.h5'
# plot_training(save_path, history_bs32_ep25, 'bs32_ep100',str(IMG_PROP))
# model.save(cp_save_path)
# history_bs32_ep50 = model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), batch_size=32, epochs=50, callbacks=callbacks)
# cp_save_path = './Plots_Unet/Full_Model_5ep_dice_diceloss_s512_bs32_ep50.h5'
# model.save(cp_save_path)
# # plot_training(history1, 'ep5')
# plot_training(history_bs32_ep50, 'bs32_ep50')


# # batch size 256
# ###############
# # plot_training(history2, 'ep10')
# history_bs32_ep5 = model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), batch_size=256, epochs=5, callbacks=callbacks)
# cp_save_path = './Plots_Unet/Full_Model_5ep_dice_diceloss_s512_bs256_ep5.h5'
# model.save(cp_save_path)
# # plot_training(history1, 'ep5')
# history_bs32_ep10 = model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), batch_size=256, epochs=10, callbacks=callbacks)
# cp_save_path = './Plots_Unet/Full_Model_5ep_dice_diceloss_s512_bs256_ep10.h5'
# model.save(cp_save_path)
# batch_size=256
# trainingsize = round(all_dat * 0.7)
# print(trainingsize)
# validate_size = all_dat * 0.3
# def calculate_spe(y):
#   return int(math.ceil((1. * y) / batch_size))
# steps_per_epoch = calculate_spe(trainingsize)
# validation_steps = calculate_spe(validate_size)

# history_bs32_ep25 = model.fit(X_train, Y_train, validation_split=0.3, batch_size=batch_size, epochs=25, callbacks=callbacks)
# cp_save_path = './Plots_Unet/Full_Model_5ep_dice_diceloss_s512_bs256_ep100.h5'
# model.save(cp_save_path)
# plot_training(save_path, history_bs32_ep25, 'bs256_ep100',str(IMG_PROP))

# history_bs32_ep50 = model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), batch_size=256, epochs=50, callbacks=callbacks)
# cp_save_path = './Plots_Unet/Full_Model_5ep_dice_diceloss_s512_bs256_ep50.h5'
# model.save(cp_save_path)
# # plot_training(history1, 'ep5')
# plot_training(history_bs32_ep50, 'bs256_ep50', str(IMG_PROP))


