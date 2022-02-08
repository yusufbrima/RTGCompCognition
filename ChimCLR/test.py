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


if __name__ == "__main__":
     viz =  Visualize(config.figures['figpath'])
     dl =  DataLoader(datapath = config.data['file_path2'])
     dl.create_tensor_set()

     model =  ChimCLR()
     temp_inputs = keras.Input(shape=dl.input_shape)
     model(temp_inputs)
     model.compile(optimizer='adam',
          loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
          metrics=['sparse_categorical_accuracy'])

     history = model.fit(dl.train_dataset, epochs=10, validation_data=(dl.train_dataset) )

