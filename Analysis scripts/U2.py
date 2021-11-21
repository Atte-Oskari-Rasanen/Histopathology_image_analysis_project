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
import math
from datetime import date

run_date = date.today()
from keras import backend as K
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

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

# TRAIN_PATH = sys.argv[1]
# MASK_PATH = sys.argv[2]

# print(TRAIN_PATH)
# print(MASK_PATH)

# IMG_PROP = int(sys.argv[3])
# IMG_HEIGHT = int(sys.argv[3])
# IMG_WIDTH = int(sys.argv[4])

IMG_CHANNELS = 3

# TRAIN_PATH = '/home/inf-54-2020/experimental_cop/Train_H_Final/Train_by_batches/Images/'
# MASK_PATH = '/home/inf-54-2020/experimental_cop/Train_H_Final/Masks_by_batches/Masks/'

TRAIN_PATH = "/home/inf-54-2020/experimental_cop/Train_H_Final/Train_by_batches/Images/"

MASK_PATH = "/home/inf-54-2020/experimental_cop/Train_H_Final/Masks_by_batches/Masks/"

IMG_PROP = 512
IMG_PROP = int(sys.argv[1])

IMG_HEIGHT = IMG_WIDTH = IMG_PROP
# IMG_HEIGHT = int(sys.argv[3])
# IMG_WIDTH = int(sys.argv[4])
IMG_CHANNELS = 3
# batch_size = 32
batch_size = int(sys.argv[2])

# TRAIN_PATH = '/home/inf-54-2020/experimental_cop/Train_H_Final/Train_by_batches/Images/'
# MASK_PATH = '/home/inf-54-2020/experimental_cop/Train_H_Final/Masks_by_batches/Masks/'
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
# img_patch = gen_patches(img, split_width, split_height)

X_train = import_images(TRAIN_PATH, IMG_HEIGHT,IMG_WIDTH, 3)
Y_train = import_masks(MASK_PATH, IMG_HEIGHT,IMG_WIDTH)
#Normalize images

# batch_size=128
all_train_imgs = len(os.listdir(TRAIN_PATH))

def calculate_spe(y):
  return int(math.ceil((1. * y) / batch_size))
steps_per_epoch = calculate_spe(all_train_imgs)

X_train = X_train / 255.
Y_train = Y_train / 255.

X_train = X_train.astype(np.float64)
Y_train = Y_train.astype(np.float64)

print("dtype X:", X_train.dtype)
print("dtype Y:", Y_train.dtype)


print(X_train[:5])
print(Y_train[:5])



print(X_train.shape)
print(Y_train.shape)





#Build the model
inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = tf.keras.layers.Lambda(lambda x: x / 255.)(inputs)

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


#model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer=Adam(learning_rate = 1e-4), loss=[dice_coef_loss], 
              #BinaryFocalLoss(gamma=2)
              metrics=[dice_coef, recall_m, precision_m, f1_m])

model.summary()



def plot_training(save_path, history, n, IMG_PROP, bs, date):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    figname = save_path + str(n) + '_ps_' + str(IMG_PROP) + '_' + bs +'_' + str(date) + '_Plot_loss_Dice.png'
    plt.savefig(figname)
    
    acc = history.history['dice_coef']
    #acc = history.history['accuracy']
    val_acc = history.history['val_dice_coef']
    #val_acc = history.history['val_accuracy']
    
    plt.plot(epochs, acc, 'y', label='Training Dice')
    plt.plot(epochs, val_acc, 'r', label='Validation Dice')
    plt.title('Training and validation Dice')
    plt.xlabel('Epochs')
    plt.ylabel('Dice')
    plt.legend()
    plt.show()
    figname = save_path + str(n) + '_ps_' + str(IMG_PROP) + '_' + bs +'_' + str(date) + '_Plot_Accuracy_Dice.png'
    plt.savefig(figname)

    i=+1

    

print('Done.')

################################
#Modelcheckpoint
# checkpointer = tf.keras.callbacks.ModelCheckpoint(cp_save_path1, verbose=1, save_best_only=True)
# tb_cb = tf.keras.callbacks.TensorBoard(log_dir='logs', profile_batch=0)

def noop():
    pass

# tb_cb._enable_trace = noop
cp_save_path = '/home/inf-54-2020/experimental_cop/scripts/unet_models/ALL_no_gen_25ep_dice_' + str(IMG_PROP) + '_' + str(run_date) +'_lrsch_' + str(batch_size) +'.h5'

callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss',restore_best_weights=True),
        tf.keras.callbacks.TensorBoard(log_dir='logs')]


save_path = '/home/inf-54-2020/experimental_cop/scripts/plots_unet/'

bs = batch_size

epochs = 25

history = model.fit(X_train, Y_train, validation_split=0.3, steps_per_epoch=steps_per_epoch, batch_size=batch_size, epochs=epochs, callbacks=[callbacks])
# history_bs128_ep25 = model.fit(X_train, Y_train, validation_split=0.3, batch_size=batch_size, epochs=25, callbacks=callbacks)
print(history.history)

cp_save_path = '/home/inf-54-2020/experimental_cop/scripts/unet_models/ALL_no_gen_25ep_dice_' + str(IMG_PROP) + '_' + str(bs) +'_' + str(run_date) +'_lrsch_' + str(batch_size) + '.h5'
#cp_save_path = './Plots_Unet/Full_Model_5ep_dice_diceloss_ps' + str(IMG_PROP) + '_bs128_ep25.h5'
model.save(cp_save_path)

plot_training(save_path, history, 'bs128_ep25_',str(IMG_PROP), str(batch_size), str(run_date))
import pandas as pd    

unet_history_df = pd.DataFrame(history.history) 
    
with open('trad_unet_history_df' + str(IMG_PROP) + '_' +str(run_date) + '_' + str(batch_size) + '.csv', mode='w') as f:
    unet_history_df.to_csv(f)

###evaluation
test_path_p = '/home/inf-54-2020/experimental_cop/Train_H_Final/Test_set/Images/'
test_mask_p = '/home/inf-54-2020/experimental_cop/Train_H_Final/Test_set/Masks/'
# Test_ims_masks = import_kaggledata(data_path, IMG_PROP, IMG_PROP, 3)
X_test = import_images(test_path_p, IMG_PROP, IMG_PROP, 3)
Y_test = import_masks(test_mask_p, IMG_PROP, IMG_PROP)
# X_train = Test_ims_masks[0]
# Y_train = Test_ims_masks[1]

print(X_test.shape)
print(Y_test.shape)

X_test = X_test/255.
Y_test = Y_test/255.

X_test = X_test.astype(np.float64)
Y_test = Y_test.astype(np.float64)
# evaluate the model
# loss, accuracy, precision, recall, f1 = model.evaluate(X_test, Y_test, verbose=0)
results = model.evaluate(X_test, Y_test, verbose=0)
print(results)
