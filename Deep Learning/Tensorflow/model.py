import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import numpy as np
import scipy as sp
import pandas as pd
import random
import os
import math
import tqdm


file_path = "/net/projects/scratch/winter/valid_until_31_July_2022/ybrima/data/data_genre.npz"
SAMPLE_RATE = 44100
n_fft = 2048
hop_length =  512
num_mfcc=128

def normalize(X):
    return (X -  np.mean(X, axis=0))/np.std(X, axis= 0)

if __name__ == '__main__':

    data =  np.load(file_path)


    W,X,y = data['w'],data['x'],data['y']
    
    # create train/test split
    Wx =  normalize(W)
    X_train, X_test, y_train, y_test = train_test_split(Wx, y, test_size=0.3,shuffle=True)
    
    
    # build network topology
    model = keras.Sequential([

        # input layer
        keras.layers.Flatten(input_shape=(Wx.shape[1], Wx.shape[2])),

        # 1st dense layer
        keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        # 2nd dense layer
        keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        # 3rd dense layer
        keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        # output layer
        keras.layers.Dense(10, activation='softmax')
    ])

    # compile model
    optimiser = keras.optimizers.Adam(learning_rate=0.0003)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    print( model.summary())
    
    # train model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=15)
    