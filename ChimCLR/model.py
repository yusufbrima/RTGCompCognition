import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras as keras

 

def ChimCLR(input_shape,num_classes):
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Conv2D(filters=32, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)(inputs)
    x = keras.layers.MaxPool2D(padding='SAME')(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)(x)
    x = keras.layers.MaxPool2D(padding='SAME')(x)
    x = keras.layers.Conv2D(filters=128, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)(x)
    x = keras.layers.MaxPool2D(padding='SAME')(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(units=256, activation=tf.nn.relu)(x)
    x = keras.layers.Dropout(rate=0.4)(x)
    x = keras.layers.Dense(units=num_classes, activation=tf.nn.softmax)(x)
    return keras.Model(inputs=inputs, outputs=x)