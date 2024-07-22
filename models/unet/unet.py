import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses, backend

def double_conv_block(x, n_filters, kernel_size):
    # Conv2D then ReLU activation
    x = layers.Conv2D(n_filters, kernel_size, strides = 1, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
    # Conv2D then ReLU activation
    x = layers.Conv2D(n_filters, kernel_size, strides = 1, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)

    return x

def downsample_block(x, n_filters, kernel_size=3):
    f = double_conv_block(x, n_filters, kernel_size)
    p = layers.MaxPool2D(2)(f)
    #p = layers.Dropout(0.3)(p)
    p = layers.Dropout(0.1)(p)

    return f, p

def upsample_block(x, conv_features, n_filters, kernel_size=3):
    # upsample
    # Modificar stride para 3
    x = layers.Conv2DTranspose(n_filters, 2, strides=2, padding="same")(x)
    # concatenate
    x = layers.concatenate([x, conv_features])
    # dropout
    #x = layers.Dropout(0.3)(x)
    x = layers.Dropout(0.1)(x)
    # Conv2D twice with ReLU activation
    x = double_conv_block(x, n_filters, kernel_size)

    return x

def build_unet_model(input_shape:tuple, n_classes:int, initial_n_filters:int=32):
    inputs = layers.Input(shape=input_shape)
    kernel_size = 3

    # encoder: contracting path - downsample
    # 1 - downsample
    f1, p1 = downsample_block(inputs, initial_n_filters, kernel_size)
    # 2 - downsample
    f2, p2 = downsample_block(p1, initial_n_filters * 2, kernel_size)
    # 3 - downsample
    f3, p3 = downsample_block(p2, initial_n_filters * 4, kernel_size)
    # 4 - downsample
    f4, p4 = downsample_block(p3, initial_n_filters * 8, kernel_size)
    # 5 - downsample
    f5, p5 = downsample_block(p4, initial_n_filters * 16, kernel_size)

    # 6 - bottleneck
    bottleneck = double_conv_block(p5, initial_n_filters * 32, kernel_size)

    # decoder: expanding path - upsample
    # 7 - upsample
    u7 = upsample_block(bottleneck, f5, initial_n_filters * 16, kernel_size)
    # 8 - upsample
    u8 = upsample_block(u7, f4, initial_n_filters * 8, kernel_size)
    # 9 - upsample
    u9 = upsample_block(u8, f3, initial_n_filters * 4, kernel_size)
    # 10 - upsample
    u10 = upsample_block(u9, f2, initial_n_filters * 2, kernel_size)
    # 11 - upsample
    u11 = upsample_block(u10, f1, initial_n_filters, kernel_size)

    # outputs
    #outputs = layers.Conv2D(n_classes, (1, 1), padding="same", activation = "softmax")(u11)
    outputs = layers.Dense(n_classes, activation='softmax')(u11)

    # unet model with Keras Functional API
    unet_model = tf.keras.Model(inputs, outputs, name="U-Net")

    return unet_model

def UNet(shape:tuple=(256, 256, 1), n_classes:int=2, backbone:str="densenet121", backbone_weights:str="imagenet"):
    backnet = None
    model_input = layers.Input(shape=shape)
    x = None

    if shape[-1] == 1:
        # Mapping non-rbg data to rgb -> that's because of the backbone restrictions
        model_input = layers.Conv2D(3, (1,1))(model_input)

    if backbone == "densenet121":
        backnet = keras.applications.DenseNet121(weights=backbone_weights, include_top=False, input_tensor=model_input)
        x = backnet.get_layer("conv3_block12_concat").output
    else:
        x = model_input

    unet = build_unet_model(shape, n_classes)(x)

    return keras.Model(inputs=model_input, outputs=unet, name="UNet")

import matplotlib.pyplot as plt
from skimage import io, transform, util
import numpy as np
import random
import math
import datetime
import glob
import os
import pickle
import math
from tqdm import tqdm

print(tf.config.list_physical_devices('GPU'))