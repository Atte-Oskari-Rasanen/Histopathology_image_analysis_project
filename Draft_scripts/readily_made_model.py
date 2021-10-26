#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 12:41:25 2021

@author: atte
"""
import tensorflow as tf
import os
import random
import numpy as np
 
from tqdm import tqdm 
import pickle

from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import re

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import glob
import cv2
import pickle

from keras.models import Sequential, Model
from keras.layers import Conv2D
import os

from tensorflow import keras
from skimage.filters.thresholding import _cross_entropy

model = keras.models.load_model('/home/inf-54-2020/experimental_cop/model-dsbowl2018-2.h5')
