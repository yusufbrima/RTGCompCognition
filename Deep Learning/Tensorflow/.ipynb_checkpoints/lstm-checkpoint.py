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


DATA_PATH = "/net/projects/scratch/winter/valid_until_31_July_2022/ybrima/data/data_chimp.npz"
n_output =  13
SAMPLE_RATE = 44100
n_fft = 2048
hop_length =  512
num_mfcc=128

def normalize(X):
    return (X -  np.mean(X, axis=0))/np.std(X, axis= 0)


def build_model(input_shape):
    """Generates RNN-LSTM model
    :param input_shape (tuple): Shape of input set
    :return model: RNN-LSTM model
    """

    # build network topology
    model = keras.Sequential()

    # 2 LSTM layers
    model.add(keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(keras.layers.LSTM(64))

    # dense layer
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # output layer
    model.add(keras.layers.Dense(n_output, activation='softmax'))

    return model
def prepare_datasets(test_size, validation_size):
    """Loads data and splits it into train, validation and test sets.
    :param test_size (float): Value in [0, 1] indicating percentage of data set to allocate to test split
    :param validation_size (float): Value in [0, 1] indicating percentage of train set to allocate to validation split
    :return X_train (ndarray): Input training set
    :return X_validation (ndarray): Input validation set
    :return X_test (ndarray): Input test set
    :return y_train (ndarray): Target training set
    :return y_validation (ndarray): Target validation set
    :return y_test (ndarray): Target test set
    """

    # load data
    data =  np.load(DATA_PATH)


    X,y = data['x'],data['y']

    Wx =  normalize(X)


    # create train, validation and test split
    X_train, X_test, y_train, y_test = train_test_split(Wx, y, test_size=test_size,shuffle=True)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size,shuffle=True)

    return X_train, X_validation, X_test, y_train, y_validation, y_test


if __name__ == '__main__':
    # get train, validation, test splits
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)

    # create network
    input_shape = (X_train.shape[1], X_train.shape[2]) # 130, 13

    model = build_model(input_shape)

    # compile model
    optimiser = keras.optimizers.Adam(learning_rate=0.0003)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    print( model.summary())
    
    # train model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=15)
    