import os
import numpy as np
import keras
import pickle
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout, Reshape, UpSampling2D, LeakyReLU, GlobalAveragePooling2D
from keras import regularizers
from keras import optimizers
from PIL import Image
from functools import partial
import sys


def get_cifar10_net(labels_num, input_shape = (32, 32, 3)):
    h = input_shape[0]
    w = input_shape[1]
    d = input_shape[2]
    
    model = Sequential()
    model.add(Conv2D(128, (3,3), padding='same', input_shape=(h, w, d)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3,3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3,3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))
     
    model.add(Conv2D(256, (3,3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3,3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3,3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))
     
    model.add(Conv2D(512, (3,3), padding='valid'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (1,1), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (1,1), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(GlobalAveragePooling2D())
     
    model.add(Dense(labels_num, activation='softmax'))
    
    return model

    
    
    
    