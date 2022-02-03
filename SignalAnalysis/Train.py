import wandb
from wandb.keras import WandbCallback
import tensorflow.keras.backend as K
import tensorflow.keras as keras
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,precision_score,confusion_matrix,accuracy_score
from sklearn.decomposition import  PCA,TruncatedSVD,FastICA
from tensorflow.keras.applications import VGG16, ResNet50,VGG19,ResNet50V2,EfficientNetB0
from sklearn.preprocessing import StandardScaler
from Preprocess import *
from Model import make_model,build_model
import IPython.display as ipd
import matplotlib.pyplot as plt
import librosa.display
import soundfile as sf
from pathlib import *
import pandas as pd
import numpy as np
import librosa
import glob
import shutil
import os
import cv2

wandb.init(project="Baseline", entity="ybrima")

# INPUT_PATH = Path('/net/store/cbc/projects/Pan troglodytes/audio_PH_dataset/PH/good_data/')
INPUT_PATH = Path('/net/store/cbc/projects/Pan troglodytes/audio_PH_dataset/PH/good_data/')
file_path =  '/net/store/cv/users/ybrima/scratch/data/Luango_Speaker/clip_loango.npz'
file_path2 =  '/net/store/cv/users/ybrima/scratch/data/Luango_Speaker/clip_loango_resized.npz'
FRAME_LENGHT =  1024
SAMPLE_RATE =  44100
HOP_LENGTH =  512

CLASSES =  get_classes(INPUT_PATH)


if __name__ == '__main__':
    """
        This code allows us to preprocess the dataset using varied padding techniques to make the input sequence into fixed length
     """
    data =  np.load(file_path, allow_pickle=True)
    X = data['x']
    y = data['y']
    Z =  data['z']
    CLASSES = list(data['c'])

    X,y,Z = build_dataset(INPUT_PATH,keepdims=False,crop_dims= (32,32),outpath=file_path2 )

    input_shape =  (X.shape[1],X.shape[2],1)
    output_nums =  len(CLASSES)
    X = X[...,np.newaxis]

    X = (X -  X.mean(axis=0))/ X.std(axis=0)
    X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.1, shuffle=True)

    model =  build_model(input_shape, output_nums)


    wandb.config = {
        "learning_rate": 0.003,
        "epochs": 100,
        "batch_size": 8
        }

    history =  model.fit(x=X_train,y=y_train, batch_size=2, epochs=20, validation_split=0.1,verbose=1,callbacks=[WandbCallback()] )

    print(model.evaluate(x=X_test,y=y_test))
    