import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras as keras
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,precision_score,confusion_matrix,accuracy_score
from tensorflow.keras.applications import VGG16,VGG19, ResNet50,InceptionV3,Xception,ResNet101,InceptionResNetV2
from sklearn.preprocessing import StandardScaler
import IPython.display as ipd
import matplotlib.pyplot as plt
import librosa.display
from Visualize import *
from Preprocess import *
import soundfile as sf
from pathlib import *
import pandas as pd
import numpy as np
import librosa
import glob
import shutil
import os
import cv2



BASE_APTH  = '/net/store/cv/users/ybrima/RTGCompCog/Experiments/'

INPUT_PATH = Path('./Data')
file_path =  './Data/clip_loango.npz'
file_path2 =  './Data/clip_loango_resized.npz'
FRAME_LENGHT =  1024
SAMPLE_RATE =  44100
HOP_LENGTH =  512

CLASSES =  get_classes(INPUT_PATH)
MODELS = [VGG16,VGG19, ResNet50,InceptionV3,Xception,ResNet101,InceptionResNetV2]
figures = Path('./Figures/')


def make_model(Model,input_shape, output_nums):
    K.clear_session() # Clear previous models from memory.
    base_model = Model(weights=None, include_top=False, input_shape=input_shape)
    x = base_model.output
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(output_nums, activation='softmax')(x)
    model = keras.Model(inputs=base_model.input, outputs=x)
    optim_params = dict(learning_rate = 0.001,momentum = 0.9394867962846013,decay = 0.0003)
    model.compile(loss='sparse_categorical_crossentropy',optimizer=keras.optimizers.SGD(**optim_params),metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")])
    return model

def train(MODELS,input_shape,output_nums):
    history = []
    for m in MODELS:
        model =  make_model(m, input_shape, output_nums)
        performance = {f'{model.name}_loss':[],f'{model.name}_accuracy':[],f'{model.name}_val_loss':[],f'{model.name}_val_accuracy':[], f'{model.name}_test_loss':[],f'{model.name}_test_accuracy':[]}
        print(model.summary())
        print("====================================================================")

if __name__ == '__main__':
    """
        This code allows us to run experiments with different CNN architectures to test with model works best given the dataset
     """
    data =  np.load(file_path, allow_pickle=True)
    X = data['x']
    y = data['y']
    Z =  data['z']
    CLASSES = list(data['c'])

    print(X.shape)

    