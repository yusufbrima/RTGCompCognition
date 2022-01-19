import tensorflow.keras.backend as K
import tensorflow.keras as keras
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import pandas as pd
import os
from pathlib import *
import glob
import shutil
import IPython.display as ipd
%pylab
%matplotlib inline



DATA_PATH = Path('/net/store/cv/users/ybrima/scratch/data')
ZIP_PATH =  Path( DATA_PATH, os.listdir(DATA_PATH)[-1])
INPUT_PATH = Path('/net/store/cv/users/ybrima/scratch/data/archive/16000_pcm_speeches/')

CLASSES = os.listdir(INPUT_PATH)
FRAME_LENGHT =  1024
SAMPLE_RATE =  16000
HOP_LENGTH =  512
n_fft=2048
num_mfcc=13

def build_model(input_shape,output_shape):
    """ This function builds a functional model"""
    

    inputs =  keras.Input(shape=input_shape,name="input_layer")
    x =  keras.layers.Flatten()(inputs)
    
    x =  keras.layers.Dense(32,activation="relu")(x)
    x =  keras.layers.BatchNormalization()(x)
    x =  keras.layers.Dropout(0.3)(x)

    x =  keras.layers.Dense(64,activation="relu")(x)
    x =  keras.layers.BatchNormalization()(x)
    x =  keras.layers.Dropout(0.3)(x)
    
    x =  keras.layers.Dense(128,activation="relu")(x)
    x =  keras.layers.BatchNormalization()(x)
    x =  keras.layers.Dropout(0.3)(x)
    
    x =  keras.layers.Dense(64,activation="relu")(x)
    x =  keras.layers.BatchNormalization()(x)
    x =  keras.layers.Dropout(0.3)(x)
    
    x =  keras.layers.Dense(32,activation="relu")(x)
    x =  keras.layers.BatchNormalization()(x)
    x =  keras.layers.Dropout(0.3)(x)

    outputs =  keras.layers.Dense(output_shape,activation="softmax")(x)


    model =  keras.Model(inputs=inputs,outputs=outputs,name="speaker_model")
    
    model.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.RMSprop(),metrics=["accuracy"],)
    
    return model




if __name__ == '__main__':
    
    input_shape = (30,13)
    output_shape =  len(CLASSES)
    model = build_model(input_shape,output_shape)