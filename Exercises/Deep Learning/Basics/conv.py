import numpy as np
import matplotlib.pyplot as plt 
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import tensorflow.keras.backend as K 


""" We are setting global constants """
NUM_CLASSES =  2
NUM_SAMPLES =  10000
NUM_FEATURES =  20
EPOCHS =  30
BATCH_SIZE =  32


def make_model(input_shape,output):

    model =  keras.models.Sequential([
        keras.layers.Dense(32, input_shape=input_shape, kernel_regularizer=keras.regularizers.L2(0.03), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2,2),strides=2, padding="same"),
        keras.layers.Dropout(0.3),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(64, kernel_size=(3,3), padding="same", activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2,2),strides=2, padding="same"),
        keras.layers.Dropout(0.3),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(128, kernel_size=(3,3), padding="same", activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2,2),strides=2, padding="same"),
        keras.layers.Dropout(0.3),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(64, kernel_size=(3,3), padding="same", activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2,2),strides=2, padding="same"),
        keras.layers.Dropout(0.3),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(32, kernel_size=(3,3), padding="same", activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2,2),strides=2, padding="same"),
        keras.layers.Dropout(0.3),
        keras.layers.BatchNormalization(),
        keras.layers.Flatten(),
        keras.layers.Dense(output, activation="softmax")

    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.003), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model

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



    # We are building a model now

    model =  make_model(input_shape=input_shape, output=NUM_CLASSES)


    history = model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), batch_size=BATCH_SIZE, epochs=EPOCHS)


    keras.models.save_model("cnn.h5")
    print("Model saved successfully")

