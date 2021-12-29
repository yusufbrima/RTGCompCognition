import tensorboard
import tensorflow as tf 
import tensorflow.keras as keras
import numpy as np 
import matplotlib.pyplot as plt 
import os 
import random

if __name__ == "__main__":

    print("Welcome to Keras Functional API")



    inputs = keras.Input(shape=(784,))

    x =  keras.layers.Dense(32,activation="relu",kernel_regularizer=keras.regularizers.l2(0.2))(inputs)

    x =   keras.layers.Dense(64,activation="relu",kernel_regularizer=keras.regularizers.l2(0.2))(x)
    x =  keras.layers.Dropout(0.3)(x)
    outputs =  keras.layers.Dense(10,activation="softmax")(x)

    model =  keras.Model(inputs=inputs,outputs=outputs,name="mnist")
    print(model.summary())

    keras.utils.plot_model(model,"reports/model.png",show_shapes=True)


    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.reshape(60000, 784).astype("float32") / 255
    x_test = x_test.reshape(10000, 784).astype("float32") / 255


    model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.RMSprop(),
    metrics=["accuracy"],
    )

    history = model.fit(x_train, y_train, batch_size=64, epochs=2, validation_split=0.2)

    test_scores = model.evaluate(x_test, y_test, verbose=2)
    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])