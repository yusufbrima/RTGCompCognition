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
from train import train_model
import config


tf.random.set_seed(42)

MODELS = [VGG16,VGG19, ResNet50,InceptionV3,Xception]
def normalize(x):
    return (x -  x.mean(axis = 0, keepdims =  True))/ x.std(axis = 0, keepdims = True)


if __name__ == "__main__":
     viz =  Visualize(config.figures['figpath'])
     dl =  DataLoader(datapath = config.data['file_path2'])
     dl.load()

     dl.create_tensor_set()
     df,dft = train_model(MODELS,dl,dl.input_shape,len(dl.CLASSES))

     print(df.head())
     # df.to_csv('./Data/model_results_train.csv', index = False)
     # dft.to_csv('./Data/model_results_test.csv', index = False)

