from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from utils import Visualize
from dataloader import DataLoader
import tensorflow as tf
import tensorflow.keras as keras
from model import ChimCLR
import config

tf.random.set_seed(42)

def create_model(input_shape,num_classes):
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

if __name__ == "__main__":
     viz =  Visualize(config.figures['figpath'])
     dl =  DataLoader(datapath = config.data['file_path2'])
     dl.create_tensor_set()

     # model =  ChimCLR()
     # temp_inputs = keras.Input(shape=dl.input_shape)
     # model(temp_inputs)

     model = ChimCLR(dl.input_shape,config.model['num_classes'])
     model.compile(optimizer='adam',
          loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
          metrics=['sparse_categorical_accuracy'])
     tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)
     print(model.summary())
     history = model.fit(dl.train_dataset, epochs=config.hyperparams['EPOCHS'], validation_data=(dl.test_dataset) )

