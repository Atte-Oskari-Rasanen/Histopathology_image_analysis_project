#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 16:56:35 2021

@author: atte
"""

import tensorflow as tf
import os
import numpy as np
from tqdm import tqdm 
from keras.utils import normalize
from skimage.io import imread, imshow
from skimage.transform import resize
seed = 42
np.random.seed = seed

IMG_WIDTH = 244
IMG_HEIGHT = 244
IMG_CHANNELS = 3

TRAIN_IMG_DIR = "/home/inf-54-2020/experimental_cop/Train_H_Final/Aug_Img/"
M_TRAIN_IMG_DIR = "/home/inf-54-2020/experimental_cop/Train_H_Final/Aug_Mask/"

#TRAIN_IMG_DIR = '/home/atte/kansio/img/'
#M_TRAIN_IMG_DIR ='/home/atte/kansio/img_mask/'

VAL_IMG_DIR = "/home/inf-54-2020/experimental_cop/Val_H_Final/Images/"
M_VAL_IMG_DIR = "/home/inf-54-2020/experimental_cop/Val_H_Final/Masks/"
TRAIN_PATH = '/cephyr/NOBACKUP/groups/snic2021-23-496/kaggle_data/'


X_train=[]
Y_train=[]

#use train_ids for the loops which take in the kaggle data. The formatting differs from 
#our own augmented, sliced, data files

#def import_kaggle_data(TRAIN_PATH):
train_ids = next(os.walk(TRAIN_PATH))[1] #returns all sub dirs found within this dir 
m_train_ids = next(os.walk(TRAIN_PATH))[1] #returns all sub dirs found within this dir 

#test_ids = next(os.walk(VAL_IMG_DIR))[1]
no_of_files = len(train_ids)
no_of_masks = len(m_train_ids)
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):   
    path = TRAIN_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]  
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train.append(img)  #Fill empty X_train with values from img
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',  
                                      preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)  
            
    Y_train.append(mask)   
    #return X_train, Y_train


n1 = 0
n2 = 0
print('starting the loops...')

img_dir_id = [] #list of dir ids containing patches of the certain image
ind_im_ids = [] #create an empty list for the ids of the individual images found in the subdir
n1 = 0
for root, subdirectories, files in sorted(os.walk(TRAIN_IMG_DIR)):
    #print(root)
    for subdirectory in subdirectories:
        file_path = os.path.join(root, subdirectory)
        #print(subdirectory)
        for f in os.listdir(file_path):
            if f.endswith('.png'):
                #print(f)
                img_path=file_path + '/' + f   #create first of dic values, i.e the path
                #print(img_path)
                #print(img_path)
                #imagename=ntpath.basename(imagepath)#take the name of the file from the path and save it
                img = imread(img_path)[:,:,:IMG_CHANNELS]
                img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
                #X_train[n1] = img  #Fill empty X_train with values from img
                X_train.append(img)
                #print(str(n1) + ' one loop of X_train done!')
                n1 += 1
   
X_train=np.array(X_train)
np.save('/home/inf-54-2020/experimental_cop/scripts/X_train_size244.npy', X_train)

print('Images saved into array!')
n2 = 0
for root, subdirectories, files in sorted(os.walk(M_TRAIN_IMG_DIR)):
    #print(root)
    for subdirectory in subdirectories:
        file_path = os.path.join(root, subdirectory)
        #print(subdirectory)
        for m in os.listdir(file_path):
            if m.endswith('.png'):
                #print(f)
                img_path=file_path + '/' + m   #create first of dic values, i.e the path
                #print(img_path)
                #print(img_path)
                #imagename=ntpath.basename(imagepath)#take the name of the file from the path and save it
                img = imread(img_path)[:,:,:1]
                img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
                #X_train[n1] = img  #Fill empty X_train with values from img
                Y_train.append(img)
                #print(str(n1) + ' one loop of Y_train done!')
                n1 += 1

            else:
                continue
Y_train=np.array(Y_train)

np.save('/home/inf-54-2020/experimental_cop/scripts/Y_train_size244.npy', Y_train)

print('masks saved into array!')
    
#convert X and Y train into numpy arrays
X_train=np.array(X_train)
print('X_train:')
print(X_train.shape)
print(X_train.size)
Y_train=np.array(Y_train)
print('Y_train:')
print(X_train.shape)
print(X_train.size)


# test images
#X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
X_test=[]
sizes_test = []
n3 = 0
for root, subdirectories, files in tqdm(os.walk(VAL_IMG_DIR)): #tqdm shows the progress bar of the for loop
    #print(root)
    for subdirectory in subdirectories:
    #    print(subdirectory)
        file_path = os.path.join(root, subdirectory)
     #   print(file_path)
        for f in os.listdir(file_path):
            if not f.endswith('.tif'):
                continue
            img_path=file_path + '/' + f   #create first of dic values, i.e the path
            #print(img_path)
            #imagename=ntpath.basename(imagepath)#take the name of the file from the path and save it
            img = imread(img_path)[:,:,:IMG_CHANNELS]
            img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
            X_test.append(img)
            #print(' loop of X_test done!')
X_test = np.array(X_test)
np.save('/home/inf-54-2020/experimental_cop/scripts/X_test_size244.npy', X_test)

print('Test files saved into array!')
# X_mask = []
# for root, subdirectories, files in tqdm(os.walk(M_VAL_IMG_DIR)): #tqdm shows the progress bar of the for loop
#     #print(root)
#     for subdirectory in subdirectories:
#         print(subdirectory)
#         file_path = os.path.join(root, subdirectory)
#         print(file_path)
#         for f in os.listdir(file_path):
#             img_path=file_path + '/' + f   #create first of dic values, i.e the path
#             #print(img_path)
#             #imagename=ntpath.basename(imagepath)#take the name of the file from the path and save it
#             img = imread(img_path)[:,:,:IMG_CHANNELS]
#             sizes_test.append([img.shape[0], img.shape[1]])
#             img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
#             X_mask.append(img)

# =============================================================================
# print('Resizing test images') 
# for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
#     print(n)
#     print(id_)
#     path = VAL_IMG_DIR + id_
#     img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
#     sizes_test.append([img.shape[0], img.shape[1]])
#     img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
#     X_test[n] = img
# =============================================================================

print('Done!')

print('lengths of X_ train and Y_Train: ')
print(len(X_train))
print(len(Y_train))
def unet(X_train, Y_train, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    print('Building the model...')
    X_train = np.expand_dims(normalize(np.array(X_train), axis=1),3)
    Y_train = np.expand_dims(normalize(np.array(Y_train), axis=1),3)
    #Y_train = np.expand_dims(normalize(np.array(Y_train), axis=1),3)
    
    #Build the model
    inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)
    
    #Contraction path
    c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
    
    c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
     
    c3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
     
    c4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
     
    c5 = tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    #Expansive path 
    u6 = tf.keras.layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
     
    u7 = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
     
    u8 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
     
    u9 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    c9 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
     
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
     
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    
    ################################
    #Modelcheckpoint
    cp_save_path = "/home/inf-54-2020/experimental_cop/scripts/Model_s512.h5"
    model.save(cp_save_path)
    checkpointer = tf.keras.callbacks.ModelCheckpoint(cp_save_path, verbose=1, save_best_only=True)
    #model.save_weights(cp_save_path)
    print('Model built and saved')
    
    callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
            tf.keras.callbacks.TensorBoard(log_dir='logs')]
    
    results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=50, use_multiprocessing=True, callbacks=callbacks)
    return results

####################################



# idx = random.randint(0, len(X_train))


# preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
# preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
# preds_test = model.predict(X_test, verbose=1)

 
# preds_train_t = (preds_train > 0.5).astype(np.uint8)
# preds_val_t = (preds_val > 0.5).astype(np.uint8)
# preds_test_t = (preds_test > 0.5).astype(np.uint8)


# saved_path1 = '/home/inf-54-2020/experimental_cop/saved_images/test.png'
# saved_path2 = '/home/inf-54-2020/experimental_cop/saved_images/test_mask.png'
# saved_path3 = '/home/inf-54-2020/experimental_cop/saved_images/test_pred.png'

# # Perform a sanity check on some random training samples
# ix = random.randint(0, len(preds_train_t))
# imshow(X_train[ix])
# plt.savefig(saved_path1)
# #plt.show()
# imshow(np.squeeze(Y_train[ix]))
# plt.savefig(saved_path2)
# #plt.show()
# imshow(np.squeeze(preds_train_t[ix]))
# plt.savefig(saved_path3)
# #plt.show()
# saved_path1 = '/home/inf-54-2020/experimental_cop/saved_images/test_Xtrain.png'
# saved_path2 = '/home/inf-54-2020/experimental_cop/saved_images/test_Ytrain.png'
# saved_path3 = '/home/inf-54-2020/experimental_cop/saved_images/test_pred2.png'


# # Perform a sanity check on some random validation samples
# ix = random.randint(0, len(preds_val_t))
# imshow(X_train[int(X_train.shape[0]*0.9):][ix])
# plt.savefig(saved_path1)

# #plt.show()
# imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]))
# plt.savefig(saved_path2)

# #plt.show()
# imshow(np.squeeze(preds_val_t[ix]))
# plt.savefig(saved_path3)

#plt.show()
