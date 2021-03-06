#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 15:21:23 2021

@author: atte
"""
import tensorflow as tf
import os
import random
import numpy as np
 
from tqdm import tqdm 
import pickle
from keras.utils import normalize
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import re
from tensorflow import keras

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
sizes_test = []
TRAIN_IMG_DIR = "/home/inf-54-2020/experimental_cop/Train_H_Final/Aug_Img/"
M_TRAIN_IMG_DIR = "/home/inf-54-2020/experimental_cop/Train_H_Final/Aug_Mask/"

#TRAIN_IMG_DIR = '/home/atte/kansio/img/'
#M_TRAIN_IMG_DIR ='/home/atte/kansio/img_mask/'

VAL_IMG_DIR = "/home/inf-54-2020/experimental_cop/Val_H_Final/Orginal_unpatched/"
print('Starting the script!')
# X_train = []
# Y_train = []
# n1 = 0
# n2 = 0
# print('starting the loops...')

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
#                 print(str(n1) + ' one loop of X_train done!')
#                 n1 += 1
   
# X_train=np.array(X_train)
# np.save('/home/inf-54-2020/experimental_cop/scripts/X_train_size128_Unet.npy', X_train)

# #np.save('/home/inf-54-2020/experimental_cop/scripts/X_train_size100.npy', X_train)
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
#                 print(str(n1) + ' one loop of Y_train done!')
#                 n1 += 1
   
# Y_train=np.array(Y_train)
# np.save('/home/inf-54-2020/experimental_cop/scripts/Y_train_size128_Unet.npy', Y_train)

# Y_train=np.array(Y_train)
# X_test=[]

# for root, subdirectories, files in tqdm(os.walk(VAL_IMG_DIR)): #tqdm shows the progress bar of the for loop
#     #print(root)
#     for subdirectory in subdirectories:
#     #    print(subdirectory)
#         file_path = os.path.join(root, subdirectory)
#       #   print(file_path)
#         for f in os.listdir(file_path):
#             if not f.endswith('.tif'):
#                 continue
#             img_path=file_path + '/' + f   #create first of dic values, i.e the path
#             #print(img_path)
#             #imagename=ntpath.basename(imagepath)#take the name of the file from the path and save it
#             img = imread(img_path)[:,:,:IMG_CHANNELS]
#             sizes_test.append([img.shape[0], img.shape[1]])
#             img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
#             X_test.append(img)
#             print('one loop of X_test done!')
# X_test = np.array(X_test)
# np.save('/home/inf-54-2020/experimental_cop/scripts/X_test_size128_Unet.npy', X_test)

# print('X_test saved!')


X_train = np.load('/home/inf-54-2020/experimental_cop/scripts/X_train_size128_Unet.npy')
Y_train = np.load('/home/inf-54-2020/experimental_cop/scripts/Y_train_size128_Unet.npy')
X_test = np.load('/home/inf-54-2020/experimental_cop/scripts/X_test_size128_Unet.npy')
print('lengths of X_ train and Y_Train: ')
print(len(X_train))
print(len(Y_train))

print('Building the model...')
X_train = np.expand_dims(normalize(np.array(X_train), axis=1),3)
Y_train = np.expand_dims(normalize(np.array(Y_train)),3) / 255

#Build the model
inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

#Contraction path
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
c1 = tf.keras.layers.Dropout(0.5)(c1)
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.5)(c2)
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
c8 = tf.keras.layers.Dropout(0.5)(c8)
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
 
u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.5)(c9)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
 
outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
 #changed the dropout of 0.1 to 0.5
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

################################
#Modelcheckpoint
cp_save_path = "/home/inf-54-2020/experimental_cop/scripts/New_model_bs128.h5"
model.save(cp_save_path)
#model.save(cp_save_path)
checkpointer = tf.keras.callbacks.ModelCheckpoint(cp_save_path, verbose=1, save_best_only=True)

#pickle.dump(model, open('Model_Unet.pickle','wb'))
print('Model built and saved')


callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='logs')]

results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=128, epochs=200, use_multiprocessing=True, callbacks=callbacks)

####################################
#plot the training and validation accuracy and loss at each epoch
loss = results.history['loss']
val_loss = results.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# saved_path1 = '/home/inf-54-2020/experimental_cop/saved_images/test.png'

# acc = history.history['acc']
# val_acc = history.history['val_acc']
# plt.plot(epochs, acc, 'y', label='Training acc')
# plt.plot(epochs, val_acc, 'r', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()
# plt.savefig(saved_path3)


#model = keras.models.load_model(cp_save_path)

idx = random.randint(0, len(X_train))

#take the model and predict on random images
preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

 
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

saved_path1 = '/home/inf-54-2020/experimental_cop/saved_images/test.png'
saved_path2 = '/home/inf-54-2020/experimental_cop/saved_images/test_mask.png'
saved_path3 = '/home/inf-54-2020/experimental_cop/saved_images/test_pred.png'

# Perform a sanity check on some random training samples
ix = random.randint(0, len(preds_train_t))
imshow(X_train[ix])
plt.savefig(saved_path1)
#plt.show()
imshow(np.squeeze(Y_train[ix]))
plt.savefig(saved_path2)
#plt.show()
imshow(np.squeeze(preds_train_t[ix]))
plt.savefig(saved_path3)
#plt.show()
saved_path1 = '/home/inf-54-2020/experimental_cop/saved_images/test_Xtrain.png'
saved_path2 = '/home/inf-54-2020/experimental_cop/saved_images/test_Ytrain.png'
saved_path3 = '/home/inf-54-2020/experimental_cop/saved_images/test_pred2.png'


# Perform a sanity check on some random validation samples
ix = random.randint(0, len(preds_val_t))
imshow(X_train[int(X_train.shape[0]*0.9):][ix])
plt.savefig(saved_path1)

#plt.show()
imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]))
plt.savefig(saved_path2)

#plt.show()
imshow(np.squeeze(preds_val_t[ix]))
plt.savefig(saved_path3)

#plt.show()
