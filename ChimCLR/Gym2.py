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
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from tensorflow.keras.applications import VGG16,VGG19, ResNet50,InceptionV3,Xception
from model import ChimCLR,make_model, autoencoder
from train import train_model,train_model_variable_epochs,train_model_variable_batchsize,train_model_variable_sequence,train_model_variable_representation
import config
import warnings
warnings.filterwarnings("ignore")

tf.random.set_seed(42)

MODELS = [VGG16,VGG19, ResNet50,Xception]

if __name__ == "__main__":
     viz =  Visualize(config.figures['figpath'])
    #  dl =  DataLoader(datapath = config.data['file_path'],keepdims=True,make=True)
    #  dl.load()

    #  dl.create_tensor_set()
     dft = train_model_variable_sequence(MODELS)

    # print(df.head())
    # df.to_csv('./Data/model_results_train.csv', index = False)
     dft.to_csv('./Data/model_results_variable_test_time.csv', index = False)
    # dft = train_model_variable_representation(MODELS)
    # dft.to_csv('./Data/model_results_test_variable_representation.csv', index = False)

