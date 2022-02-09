import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras as keras


def autoencoder(input_shape):
    encoder = keras.models.Sequential()
    encoder.add(keras.layers.Conv2D(32, 3, strides=1, padding='same', activation='relu', input_shape=input_shape))
    encoder.add(keras.layers.MaxPooling2D(2, strides=2))
    encoder.add(keras.layers.BatchNormalization())
    encoder.add(keras.layers.Conv2D(64, 3, strides=1, padding='same', activation='relu'))
    encoder.add(keras.layers.MaxPooling2D(2, strides=2))
    encoder.add(keras.layers.BatchNormalization())
    encoder.add(keras.layers.Conv2D(128, 3, strides=1, padding='same', activation='relu'))
    encoder.add(keras.layers.MaxPooling2D(2, strides=2))
    encoder.add(keras.layers.BatchNormalization())
    encoder.add(keras.layers.Conv2D(64, 3, strides=1, padding='same', activation='relu'))
    encoder.add(keras.layers.MaxPooling2D(2, strides=2))
    encoder.add(keras.layers.BatchNormalization())
    encoder.add(keras.layers.Conv2D(32, 3, strides=1, padding='same', activation='relu'))
    encoder.add(keras.layers.Flatten())



    decoder = keras.models.Sequential()

    decoder.add(keras.layers.Reshape((8, 8, 32), input_shape=encoder.output.shape[1:]))
    decoder.add(keras.layers.Conv2D(32, 3, strides=1, padding='same', activation='relu', input_shape=encoder.output.shape[1:]))
    decoder.add(keras.layers.UpSampling2D(2))
    encoder.add(keras.layers.BatchNormalization())
    decoder.add(keras.layers.Conv2D(64, 3, strides=1, padding='same', activation='relu'))
    decoder.add(keras.layers.UpSampling2D(2))
    encoder.add(keras.layers.BatchNormalization())
    decoder.add(keras.layers.Conv2D(128, 3, strides=1, padding='same', activation='relu'))
    decoder.add(keras.layers.UpSampling2D(2))
    encoder.add(keras.layers.BatchNormalization())
    decoder.add(keras.layers.Conv2D(1, 3, strides=1, padding='same', activation='relu'))
    decoder.add(keras.layers.UpSampling2D(2))
    conv_autoencoder = keras.Model(inputs=encoder.input, outputs=decoder(encoder.outputs))
    optim_params = dict(learning_rate = 0.003,momentum = 0.9394867962846013,decay = 0.0003)
    conv_autoencoder.compile(optimizer=keras.optimizers.SGD(**optim_params), loss=keras.losses.mean_squared_error, metrics = ['accuracy'])
    return encoder, decoder, conv_autoencoder

def ChimCLR(input_shape,num_classes):
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Conv2D(filters=32, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)(inputs)
    x = keras.layers.MaxPool2D(padding='SAME')(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)(x)
    x = keras.layers.MaxPool2D(padding='SAME')(x)
    x = keras.layers.Conv2D(filters=128, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)(x)
    x = keras.layers.MaxPool2D(padding='SAME')(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(units=256, activation=tf.nn.relu)(x)
    x = keras.layers.Dropout(rate=0.4)(x)
    x = keras.layers.Dense(units=128, activation=tf.nn.relu)(x)
    x = keras.layers.Dropout(rate=0.4)(x)
    x = keras.layers.Dense(units=64, activation=tf.nn.relu)(x)
    x = keras.layers.Dropout(rate=0.4)(x)
    x = keras.layers.Dense(units=32, activation=tf.nn.relu)(x)
    x = keras.layers.Dropout(rate=0.4)(x)
    x = keras.layers.Dense(units=num_classes, activation=tf.nn.softmax)(x)
    return keras.Model(inputs=inputs, outputs=x)

def make_model(input_shape, num_classes):
    K.clear_session() # Clear previous models from memory.
    base_model = keras.applications.VGG16(weights=None, include_top=False, input_shape=input_shape)
    x = base_model.output
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs=base_model.input, outputs=x)
    optim_params = dict(learning_rate = 0.003,momentum = 0.9394867962846013,decay = 0.0003)
    model.compile(loss='sparse_categorical_crossentropy',optimizer=keras.optimizers.SGD(**optim_params),metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")]) #keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='sparse_top_5_categorical_accuracy')
    return model

def build_model(Model,input_shape, num_classes):
    K.clear_session() # Clear previous models from memory.
    base_model = Model(weights=None, include_top=False, input_shape=input_shape)
    x = base_model.output
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs=base_model.input, outputs=x, name = base_model.name)
    optim_params = dict(learning_rate = 0.001,momentum = 0.9394867962846013,decay = 0.0003)
    model.compile(loss='sparse_categorical_crossentropy',optimizer=keras.optimizers.SGD(**optim_params),metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")])
    return model