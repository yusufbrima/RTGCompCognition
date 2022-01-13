import tensorboard
import tensorflow as tf 
import tensorflow.keras as keras
import numpy as np 
import matplotlib.pyplot as plt 
import os 
import random

if __name__ == "__main__":

    input_dims = (784,)

    inputs =  keras.Input(shape=(input_dims),name="input_layer")
    x =  keras.layers.Flatten()(inputs)
    x =  keras.layers.Dense(32,activation="relu", kernel_regularizer=keras.regularizers.l2(0.03))(x)
    x =  keras.layers.BatchNormalization()(x)
    x =  keras.layers.Dropout(0.3)(x)

    x =  keras.layers.Dense(64,activation="relu", kernel_regularizer=keras.regularizers.l2(0.03))(x)
    x =  keras.layers.BatchNormalization()(x)
    x =  keras.layers.Dropout(0.3)(x)

    x =  keras.layers.Dense(128,activation="relu", kernel_regularizer=keras.regularizers.l2(0.03))(x)
    x =  keras.layers.BatchNormalization()(x)
    x =  keras.layers.Dropout(0.3)(x)

    x =  keras.layers.Dense(64,activation="relu", kernel_regularizer=keras.regularizers.l2(0.03))(x)
    x =  keras.layers.BatchNormalization()(x)
    x =  keras.layers.Dropout(0.3)(x)

    x =  keras.layers.Dense(32,activation="relu", kernel_regularizer=keras.regularizers.l2(0.03))(x)
    x =  keras.layers.BatchNormalization()(x)
    x =  keras.layers.Dropout(0.3)(x)

    outputs =  keras.layers.Dense(10,activation="softmax")(x)


    model =  keras.Model(inputs=inputs,outputs=outputs,name="mnist_model")

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.003), loss="sparse_categorical_crossentropy",metrics=["accuracy",keras.metrics.AUC()])
    

    # model.fit(x_train,y_train,validation_split=0.2,batch_size=32,epochs=15)

    keras.utils.plot_model(model,"model.png",show_shapes=True)




