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
from datetime import datetime 

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
from Models_unet_import import *
from import_images_masks_patches import *


from datetime import date

run_date = date.today()
model_path = "/home/inf-54-2020/experimental_cop/scripts/unet_models/"


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

TRAIN_PATH = "/home/inf-54-2020/experimental_cop/Train_H_Final/Train_by_batches/Images/"

MASK_PATH = "/home/inf-54-2020/experimental_cop/Train_H_Final/Masks_by_batches/Masks/"

TRAIN_VAL_PATH = "/home/inf-54-2020/experimental_cop/Train_H_Final/Train_by_batches/Val_Images/"

MASK_VAL_PATH = "/home/inf-54-2020/experimental_cop/Train_H_Final/Masks_by_batches/Val_Masks/"

print(TRAIN_PATH)
print(MASK_PATH)

IMG_PROP = 512
IMG_HEIGHT = IMG_WIDTH = IMG_PROP
# IMG_HEIGHT = int(sys.argv[3])
# IMG_WIDTH = int(sys.argv[4])
IMG_CHANNELS = 3
batch_size = 32
# TRAIN_PATH = '/home/inf-54-2020/experimental_cop/Train_H_Final/Train_by_batches/Images/'
# MASK_PATH = '/home/inf-54-2020/experimental_cop/Train_H_Final/Masks_by_batches/Masks/'
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
# img_patch = gen_patches(img, split_width, split_height)

X_train = import_images(TRAIN_PATH, IMG_HEIGHT,IMG_WIDTH, 3)
Y_train = import_masks(MASK_PATH, IMG_HEIGHT,IMG_WIDTH)
X_val = import_images(TRAIN_VAL_PATH, IMG_HEIGHT,IMG_WIDTH, 3)
Y_val = import_masks(MASK_VAL_PATH, IMG_HEIGHT,IMG_WIDTH)

seed = random.seed(42)

img_data_gen_args = dict(rescale = 1/255.)

image_data_generator = ImageDataGenerator(**img_data_gen_args)
mask_data_gen_args = dict(rescale = 1/255.)

image_generator = image_data_generator.flow(X_train, 
                                            seed=seed, 
                                            batch_size=batch_size,
                                            )  #Very important to set this otherwise it returns multiple numpy arrays 
                                                                            #thinking class mode is binary.

mask_data_generator = ImageDataGenerator(**mask_data_gen_args)
mask_generator = mask_data_generator.flow(Y_train, 
                                            seed=seed, 
                                            batch_size=batch_size,
                                            # color_mode = 'grayscale',   #Read masks in grayscale
                                            )


valid_img_generator = image_data_generator.flow(X_val, 
                                                seed=seed, 
                                                batch_size=batch_size, 
                                                ) #Default batch size 32, if not specified here
valid_mask_generator = mask_data_generator.flow(Y_val, 
                                                seed=seed, 
                                                batch_size=batch_size, 
                                                # color_mode = 'grayscale',   #Read masks in grayscale
                                                )  #Default batch size 32, if not specified here


train_generator = zip(image_generator, mask_generator)
val_generator = zip(valid_img_generator, valid_mask_generator)

# X_train = import_images(TRAIN_PATH, IMG_HEIGHT,IMG_WIDTH, 3)
# Y_train = import_masks(MASK_PATH, IMG_HEIGHT,IMG_WIDTH)

#Normalize images

# batch_size=128
all_train_imgs = len(os.listdir(TRAIN_PATH))

def calculate_spe(y):
  return int(math.ceil((1. * y) / batch_size))
steps_per_epoch = calculate_spe(all_train_imgs)
epochs = 3

'''
Attention UNet
'''
input_shape = (512,512,3)
att_unet_model = Attention_UNet(input_shape)

att_unet_model.compile(optimizer=Adam(lr = 1e-2), loss=dice_coef_loss, 
              metrics=['accuracy', dice_coef])


print(att_unet_model.summary())
start2 = datetime.now() 
print('steps per epoch: ' + str(steps_per_epoch))
print('number of training files: '+ str(all_train_imgs))
# att_unet_history = att_unet_model.fit(X_train, Y_train, validation_split=0.3,
#                     verbose=1,
#                     batch_size = batch_size,
#                     shuffle=False,
#                     epochs=1)
history_att_unet = att_unet_model.fit_generator(train_generator, validation_data=val_generator, steps_per_epoch=steps_per_epoch, validation_steps=steps_per_epoch, epochs=epochs)

stop2 = datetime.now()
#Execution time of the model 
execution_time_Att_Unet = stop2-start2
print("Attention UNet execution time is: ", execution_time_Att_Unet)

att_unet_model.save(model_path + 'ALL_Attention_UNet_1epochs_Dice.h5')

#
#___________________________________________
'''
Attention Residual Unet
'''
att_res_unet_model = Attention_ResUNet(input_shape)

att_res_unet_model.compile(optimizer=Adam(lr = 1e-2), loss=dice_coef_loss, 
              metrics=['accuracy', dice_coef])


# att_res_unet_model.compile(optimizer=Adam(lr = 1e-3), loss='binary_crossentropy', 
#               metrics=['accuracy', jacard_coef])

print(att_res_unet_model.summary())


start3 = datetime.now() 
# att_res_unet_history = att_res_unet_model.fit(X_train, Y_train, validation_split=0.3,
#                     verbose=1,
#                     batch_size = batch_size,
#                     shuffle=False,
#                     epochs=1)
history_att_res_unet = att_res_unet_model.fit_generator(train_generator, validation_data=val_generator, steps_per_epoch=steps_per_epoch, validation_steps=steps_per_epoch, epochs=epochs)

stop3 = datetime.now() 

#Execution time of the model 
execution_time_AttResUnet = stop3-start3
print("Attention ResUnet execution time is: ", execution_time_AttResUnet)

att_res_unet_model.save(model_path + 'All_AttResUnet_1epochs_dice.h5')

############################################################################
# convert the history.history dict to a pandas DataFrame and save as csv for
# future plotting
import pandas as pd    
att_unet_history_df = pd.DataFrame(history_att_unet.history) 
att_res_unet_history_df = pd.DataFrame(history_att_res_unet.history) 
    
with open('att_unet_history_df.csv', mode='w') as f:
    att_unet_history_df.to_csv(f)

with open('custom_code_att_res_unet_history_df.csv', mode='w') as f:
    att_res_unet_history_df.to_csv(f)    

#######################################################################
#Check history plots, one model at a time
# history = unet_history
histories = [history_att_unet, history_att_res_unet]

plots_path =  "/home/inf-54-2020/experimental_cop/scripts/plots_unet/"

for h in histories:
    unets = [Att_Unet, Att_Res_Unet]
    i=0
    plot_name = unets[0]
    # plot the training and validation accuracy and loss at each epoch
    loss = h.history['loss']
    val_loss = h.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    figname = plots_path + unets[i] + 'Plot_loss.png'
    plt.savefig(figname)
    
    acc = h.history['accuracy']
    #acc = history.history['accuracy']
    val_acc = h.history['val_accuracy']
    #val_acc = history.history['val_accuracy']
    
    plt.plot(epochs, acc, 'y', label='Training Dice')
    plt.plot(epochs, val_acc, 'r', label='Validation Dice')
    plt.title('Training and validation Dice')
    plt.xlabel('Epochs')
    plt.ylabel('Dice')
    plt.legend()
    plt.show()
    figname = plots_path + unets[i] +'Plot_acc.png'
    i=+1
#######################################################
