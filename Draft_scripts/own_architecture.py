#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 13:42:51 2021

@author: atte
"""
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten,Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from tensorflow.keras.layers import MaxPooling2D
#from Data_import import *
#Initialize CNN and add conv. layer
print('starting...')
X_train = np.load('/home/inf-54-2020/experimental_cop/scripts/X_train_size128.npy')
Y_train = np.load('/home/inf-54-2020/experimental_cop/scripts/Y_train_size128.npy')
X_test = np.load('/home/inf-54-2020/experimental_cop/scripts/X_test_size128.npy')

shapes = X_train.shape
print(Y_train.shape)

model=Sequential()

model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(128,128,3)))

#Pooling operation
# We need to apply the pooling operation after initializing CNN. Pooling is an operation
# of down sampling of the image. The pooling layer is used to reduce the dimensions
# of the feature maps. Thus, the Pooling layer reduces the number of parameters to
# learn and reduces computation in the neural network.
model.add(MaxPooling2D(pool_size=2))

# In order to add two more convolutional layers, we need to repeat earlier steps
# with slight modification in the number of filters.
model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation ="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))

#After finishing the 3 steps, now we have pooled feature map. We are now flattening
#our output after two steps into a column. Because we need to insert this 1-D data into
#an artificial neural network layer.

# Flattening operation is converting the dataset into a 1-D array for input into 
# the next layer which is the fully connected layer. 
# The output of the flattening operation work as input for the neural network
model.add(Flatten())

#dense layers
model.add(Dense(500,activation="relu"))

model.add(Dense(3,activation="softmax"))
# The softMax activation function is used for building the output layer.
# It is used as the last activation function of a neural network to bring the 
# output of the neural network to a probability distribution over predicting classes.
#  The output of Softmax is in probabilities of each possible outcome for predicting
#  class. The probabilities sum should be one for all possible predicting classes.
callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='logs')]

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

cp_save_path = "/home/inf-54-2020/experimental_cop/scripts/Own_model.h5"
model.save(cp_save_path)

r = model.fit(X_train, Y_train, validation_split=0.1, batch_size=128, epochs=200, use_multiprocessing=True, callbacks=callbacks)
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
#plt.show()
plt.savefig('LossVal_loss.pdf')

plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
#plt.show()
plt.savefig('AccVal_acc.pdf')

#model.fit_generator(X_train,Y_train, epochs=50, steps_per_epoch=len(training_set), validation_steps=len(test_set) )

