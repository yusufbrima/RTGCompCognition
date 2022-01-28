import tensorboard
import tensorflow as tf 
import tensorflow.keras as keras
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import os 
import random


def build_model():

    encoder_input =  keras.Input(shape=(28,28,1),name="img")

    x =  keras.layers.Conv2D(16, 3,activation="relu")(encoder_input)
    x = keras.layers.Conv2D(32, 3, activation="relu")(x)
    x = keras.layers.MaxPooling2D(3)(x)
    x = keras.layers.Conv2D(32, 3, activation="relu")(x)
    x = keras.layers.Conv2D(16, 3, activation="relu")(x)
    encoder_output =  keras.layers.GlobalMaxPooling2D()(x)
    
    encoder = keras.Model(encoder_input, encoder_output, name="encoder")

    return encoder

if __name__ == "__main__":
    """  Welcome to DAG """

    X,y =  make_classification(n_samples=100000, n_features=784,n_classes=9)