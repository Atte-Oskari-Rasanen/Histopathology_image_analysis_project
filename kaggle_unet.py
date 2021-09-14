#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 18:08:17 2021

@author: atte
"""

import tensorflow as tf
import os
import random
import numpy as np
 
from tqdm import tqdm 
import sys
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib
matplotlib.use('Agg')
import cv2
import matplotlib.pyplot as plt

seed = 42
np.random.seed = seed

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

# TRAIN_PATH = '/home/inf-54-2020/experimental_cop//kaggle_data/'
# #TRAIN_PATH = sys.argv[1]

# #TEST_PATH = 'stage1_test/'
# #X_test = np.load('/home/inf-54-2020/experimental_cop/scripts/X_test_size128.npy')
# train_ids = next(os.walk(TRAIN_PATH))[1]
# #test_ids = next(os.walk(TEST_PATH))[1]

# # X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
# # Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

# X_train = []
# Y_train = []
# print('Resizing training images and masks')
# for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):   
#     path = TRAIN_PATH + id_
#     img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]  
#     img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
#     X_train.append(img)  #Fill empty X_train with values from img
#     mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
#     for mask_file in next(os.walk(path + '/masks/'))[2]:
#         mask_ = imread(path + '/masks/' + mask_file)
#         mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',  
#                                       preserve_range=True), axis=-1)
#         mask = np.maximum(mask, mask_)  
            
#     Y_train.append(mask)   

##########################

X_train = np.load('/cephyr/NOBACKUP/groups/snic2021-23-496/scripts/kd_X_train_size128.npy')
Y_train = np.load('/cephyr/NOBACKUP/groups/snic2021-23-496/scripts/kd_Y_train_size128.npy')

# X_train = np.load('/cephyr/NOBACKUP/groups/snic2021-23-496/scripts/X_train_size128.npy')
# Y_train = np.load('/cephyr/NOBACKUP/groups/snic2021-23-496/scripts/Y_train_size128.npy')

# TRAIN_IMG_DIR = sys.argv[2]
# M_TRAIN_IMG_DIR = sys.argv[3]

# img_dir_id = [] #list of dir ids containing patches of the certain image
# ind_im_ids = [] #create an empty list for the ids of the individual images found in the subdir
# n1 = 0
# for root, subdirectories, files in sorted(os.walk(TRAIN_IMG_DIR)):
#     #print(root)
#     for subdirectory in subdirectories:
#         file_path = os.path.join(root, subdirectory)
#         #print(subdirectory)
#         for f in os.listdir(file_path):
#             if f.endswith('.png'):
#                 #print(f)
#                 img_path=file_path + '/' + f   #create first of dic values, i.e the path
#                 #print(img_path)
#                 #print(img_path)
#                 #imagename=ntpath.basename(imagepath)#take the name of the file from the path and save it
#                 img = imread(img_path)[:,:,:IMG_CHANNELS]
#                 img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
#                 #X_train[n1] = img  #Fill empty X_train with values from img
#                 X_train.append(img)
#                 #print(str(n1) + ' one loop of X_train done!')
#                 n1 += 1
   
# X_train=np.array(X_train)
# #np.save('/home/inf-54-2020/experimental_cop/scripts/X_train_size128.npy', X_train)

# print('Images saved into array!')
# n2 = 0
# for root, subdirectories, files in sorted(os.walk(M_TRAIN_IMG_DIR)):
#     #print(root)
#     for subdirectory in subdirectories:
#         file_path = os.path.join(root, subdirectory)
#         #print(subdirectory)
#         for m in os.listdir(file_path):
#             if m.endswith('.png'):
#                 #print(f)
#                 img_path=file_path + '/' + m   #create first of dic values, i.e the path
#                 #print(img_path)
#                 #print(img_path)
#                 #imagename=ntpath.basename(imagepath)#take the name of the file from the path and save it
#                 img = imread(img_path)[:,:,:1]
#                 img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
#                 #X_train[n1] = img  #Fill empty X_train with values from img
#                 Y_train.append(img)
#                 #print(str(n1) + ' one loop of Y_train done!')
#                 n1 += 1

#             else:
#                 continue
# Y_train=np.array(Y_train)

# np.save('/cephyr/NOBACKUP/groups/snic2021-23-496/X_train_kagl_own_s512.npy', X_train)
# np.save('/cephyr/NOBACKUP/groups/snic2021-23-496/X_train_kagl_own_s512.npy', Y_train)

# #np.save('/home/inf-54-2020/experimental_cop/scripts/Y_train_size128.npy', Y_train)
# print(Y_train.shape)
# print(Y_train)
# print('masks saved into array!')
###########
# test images
#X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
# sizes_test = []
# print('Resizing test images') 
# for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
#     path = TEST_PATH + id_
#     img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
#     sizes_test.append([img.shape[0], img.shape[1]])
#     img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
#     X_test[n] = img

# VAL_IMG_DIR = "/home/inf-54-2020/experimental_cop/Val_H_Final/Images/"
# X_test=[]
# sizes_test = []
# n3 = 0
# print('starting...')
# for root, subdirectories, files in tqdm(os.walk(VAL_IMG_DIR)): #tqdm shows the progress bar of the for loop
#     #print(root)
#     for subdirectory in subdirectories:
#     #    print(subdirectory)
#         file_path = os.path.join(root, subdirectory)
#       #   print(file_path)
#         for f in os.listdir(file_path):
#             if not f.endswith('.png'):
#                 continue
#             img_path=file_path + '/' + f   #create first of dic values, i.e the path
#             #print(img_path)
#             #imagename=ntpath.basename(imagepath)#take the name of the file from the path and save it
#             img = imread(img_path)[:,:,:IMG_CHANNELS]
#             sizes_test.append([img.shape[0], img.shape[1]])
#             img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
#             X_test.append(img)
#             #print(' loop of X_test done!')
# X_test = np.array(X_test)
# print(X_test.shape)

# np.save('/home/inf-54-2020/experimental_cop/scripts/kd_X_test_size244.npy', X_test)
# np.save('/home/inf-54-2020/experimental_cop/scripts/kd_X_train_size244.npy', X_train)
# np.save('/home/inf-54-2020/experimental_cop/scripts/kd_Y_train_size244.npy', Y_train)

# print('Done!')

# image_x = random.randint(0, len(train_ids))
# imshow(X_train[image_x])
# #plt.show()
# imshow(np.squeeze(Y_train[image_x]))
# #plt.show()
# im_path = "/home/inf-54-2020/experimental_cop/saved_images/checkup.png"

# plt.savefig(im_path)


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
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

################################
#Modelcheckpoint
#cp_save_path = "/cephyr/NOBACKUP/groups/snic2021-23-496/kaggle_model_size244.h5"
cp_save_path = "/cephyr/NOBACKUP/groups/snic2021-23-496/scripts/kaggle_model_size128.h5"
cp_save_path = "/home/inf-54-2020/experimental_cop/scripts/working_models/kaggle_model_size128.h5"

checkpointer = tf.keras.callbacks.ModelCheckpoint(cp_save_path, verbose=1, save_best_only=True)

callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='logs')]


model.save(cp_save_path)
checkpointer = tf.keras.callbacks.ModelCheckpoint(cp_save_path, verbose=1, save_best_only=True)
#model.save_weights(cp_save_path)
print('Model built and saved, now fitting it...')
history = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=200, callbacks=callbacks)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('plot1_kaggledata128.png')

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('plot2_kaggledata128.png')


print('Done.')
####################################

# idx = random.randint(0, len(X_train))


# preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
# preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
# preds_test = model.predict(X_test, verbose=1)

 
# preds_train_t = (preds_train > 0.5).astype(np.uint8)
# preds_val_t = (preds_val > 0.5).astype(np.uint8)
# preds_test_t = (preds_test > 0.5).astype(np.uint8)


# #Perform a sanity check on some random training samples
# ix = random.randint(0, len(preds_train_t))
# imshow(X_train[ix])
# im_path = "/home/inf-54-2020/experimental_cop/saved_images/X_traintest1.png"
# cv2.imwrite(im_path, ix)
# #plt.savefig(im_path)

# #plt.show()

# imshow(np.squeeze(Y_train[ix]))
# im_path = "/home/inf-54-2020/experimental_cop/saved_images/Y_traintest1.png"
# plt.savefig(im_path)

# #plt.show()
# imshow(np.squeeze(preds_train_t[ix]))
# im_path = "/home/inf-54-2020/experimental_cop/saved_images/preds_traintest1.png"
# plt.savefig(im_path)

# #plt.show()

# #Perform a sanity check on some random validation samples
# ix = random.randint(0, len(preds_val_t))
# imshow(X_train[int(X_train.shape[0]*0.9):][ix])

# im_path = "/home/inf-54-2020/experimental_cop/saved_images/preds_traintest1.png"
# plt.savefig(im_path)

# plt.show()
# imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]))
# plt.show()
# imshow(np.squeeze(preds_val_t[ix]))
# plt.show()








