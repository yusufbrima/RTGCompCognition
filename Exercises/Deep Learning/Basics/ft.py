import numpy as np
import matplotlib.pyplot as plt 
from sklearn.datasets import make_classification,make_regression
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from tensorflow.keras.applications import vgg16
import tensorflow.keras.backend as K 


""" We are setting global constants """
NUM_CLASSES =  2
NUM_SAMPLES =  10000
NUM_FEATURES =  20
EPOCHS =  30
BATCH_SIZE =  32


if __name__ == "__main__":
    """  We are making the model workhorse here"""

    #Step 1, define a basic dataset aka fake one 
    
    X,y =  make_classification(n_samples=10000,n_features=1024,n_classes=2)

    X_train,X_valid,Y_train,Y_valid = train_test_split(X,y,test_size=0.2,random_state=np.random.seed(1992))

    X_train,X_test,Y_train,Y_test = train_test_split(X_train,Y_train,test_size=0.1,random_state=np.random.seed(1992))

    # We are reshaping the data for 2D convolution
    X_train = X_train[...,np.newaxis].reshape(-1,32,32,1)
    X_valid = X_valid[...,np.newaxis].reshape(-1,32,32,1)
    X_test = X_test[...,np.newaxis].reshape(-1,32,32,1)
    input_shape =  X_train[0].shape

    vgg16 = keras.applications.vgg16.VGG16() #weights="imagenet",classes=NUM_CLASSES,input_shape=input_shape

    print(type(vgg16))

    model = keras.models.Sequential()

    for layer in vgg16.layers:
        model.add(layer)
    
    model.layers.pop()

    for layer in model.layers:
        layer.trainable =  False 
    
    model.add(keras.layers.Dense(NUM_CLASSES, activation="softmax"))

    model.layers[0].input_shape = input_shape
    
    print(model.summary())


   





