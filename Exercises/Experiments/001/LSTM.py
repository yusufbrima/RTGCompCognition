import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler,Normalizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix, f1_score,precision_score
from sklearn.decomposition import FastICA, PCA,TruncatedSVD
import tensorflow.keras as keras
from pathlib import Path
from autoencode import ae
from preprocessing import preprocess,noise
from visualization import display 
import numpy as np
import pandas as pd
import random
import os
import math

file_path = "/net/projects/scratch/winter/valid_until_31_July_2022/ybrima/data/data_chimp_preprocessed.npz"
CLASSES = ['kub','woo','rom','uta','jac','ish','sum','jul','kin','Kuba']
SAMPLE_RATE = 44100
N_FFT = 2048
HOP_LENGTH =  512

def make_model(n_in):
    # define model
    model = keras.Sequential()
    model.add(keras.layers.LSTM(100, activation='relu', input_shape=(n_in,1)))
    model.add(keras.layers.RepeatVector(n_in))
    model.add(keras.layers.LSTM(100, activation='relu', return_sequences=True))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(1)))
    model.compile(optimizer='adam', loss='mse',metrics=['accuracy'])
    return model
if __name__ == '__main__':
    """ This script implements the auto encoder module that removes noise from the input signal"""
    data =  np.load(file_path,allow_pickle=True)
    X = data['X']
    y = data['y']
    sr = data['sr']

    # reshape input into [samples, timesteps, features]
    n_in = X.shape[1]
    X =  X[...,np.newaxis]

    model =  make_model(n_in=n_in)

    # fit model
    history = model.fit(X, X, epochs=10, verbose=1, batch_size=32)