import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow.keras  as keras
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K


def buildModel(input_shape,n_classes=2):
    K.clear_session()
    model =  keras.models.Sequential([
        keras.layers.Dense(32,input_shape=input_shape,activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.BatchNormalization(),
        # 2nd layer
        keras.layers.Dense(64,activation="relu",kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        keras.layers.BatchNormalization(),
        
        # 3rd layer
        keras.layers.Dense(128,activation="relu",kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        keras.layers.BatchNormalization(),

        # 4th layer
        keras.layers.Dense(64,activation="relu",kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        keras.layers.BatchNormalization(),

        keras.layers.Dense(2,activation="softmax"),
    ])
    model.compile(loss="sparse_categorical_crossentropy",metrics=["accuracy"],optimizer=keras.optimizers.Adam(learning_rate=0.0003))

    print(model.summary())
    return model


def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs

        :param history: Training history of model
        :return:
    """

    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()

if __name__ == '__main__':
    n_classes = 2
    X,y = make_classification(n_samples=10000, n_features=30, n_classes=n_classes)
    X_train,X_test,Y_train,Y_test =  train_test_split(X,y,test_size=0.2,shuffle=True)

    input_shape =  (X_train.shape[1],)

    model =  buildModel(input_shape,n_classes=n_classes)

    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=32, epochs=15)
    
    # plot accuracy/error for training and validation
    plot_history(history)