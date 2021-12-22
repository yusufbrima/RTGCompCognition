import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
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
    data =  np.load(file_path)


    W,X,y = data['w'],data['x'],data['y']

    Wx =  normalize(W)


    # create train, validation and test split
    X_train, X_test, y_train, y_test = train_test_split(Wx, y, test_size=test_size,shuffle=True)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size,shuffle=True)

    # add an axis to input sets
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(input_shape):
    """Generates CNN model
    :param input_shape (tuple): Shape of input set
    :return model: CNN model
    """

    # build network topology
    model = keras.Sequential()

    # 1st conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 2nd conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 3rd conv layer
    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())
    

    # flatten output and feed it into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # output layer
    model.add(keras.layers.Dense(10, activation='softmax'))

    return model

def predict(model, X, y):
    """Predict a single sample using the trained model
    :param model: Trained classifier
    :param X: Input data
    :param y (int): Target
    """

    # add a dimension to input data for sample - model.predict() expects a 4d array in this case
    X = X[np.newaxis, ...] # array shape (1, 130, 13, 1)

    # perform prediction
    prediction = model.predict(X)

    # get index with max value
    predicted_index = np.argmax(prediction, axis=1)

    print("Target: {}, Predicted label: {}".format(y, predicted_index))


if __name__ == '__main__':

    # get train, validation, test splits
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)


    # create network
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    model = build_model(input_shape)

    # compile model

    optim_params = dict(
        learning_rate = 0.0003,
        momentum = 0.9394867962846013,
        decay = 0.0001
    )

    optimiser = keras.optimizers.SGD(**optim_params)
    model.compile(optimizer=optimiser,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])


    check_point= ModelCheckpoint(
        # 'model_v2.best.h5', 
        'weighted_model_v2.best.h5', 
        monitor='val_loss', verbose=1, 
        save_best_only=True, save_weights_only=False, save_freq=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=5, verbose=1, min_lr=0.00001
    )

    early_stop= EarlyStopping(
        monitor='val_loss', 
        min_delta=0.001, patience=11, verbose=1, restore_best_weights=True
    )

    # train model
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=15,callbacks=[check_point, early_stop, reduce_lr])