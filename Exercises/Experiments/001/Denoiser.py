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


if __name__ == '__main__':
    """ This script implements the auto encoder module that removes noise from the input signal"""
    data =  np.load(file_path,allow_pickle=True)
    X = data['Z']
    y = data['y']
    sr = data['sr']
    train_data,test_data,train_label, test_label = train_test_split(X,y,test_size=0.2)
    # Normalize and reshape the data
    train_data = preprocess(train_data)
    test_data = preprocess(test_data)

    # Create a copy of the data with added noise
    noisy_train_data = noise(train_data)
    noisy_test_data = noise(test_data)
    # n = 5
    # display(train_data[:n,], noisy_train_data[:n,],sr[:n])

    input_shape = (1025, 173,1)
    print(train_data.shape)
    print(noisy_train_data.shape)
    model =  ae(input_shape)
    print(model.summary())
    # history =  model.fit(x=noisy_train_data,y=train_data, epochs=10,batch_size=128,shuffle=True,validation_split=0.2 )

    # predictions = model.predict(noisy_test_data)
    # display(test_data, predictions,sr)

