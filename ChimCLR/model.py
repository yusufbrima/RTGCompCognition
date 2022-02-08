import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras as keras
import config


 
# class ChimCLR:

#     def __init__(self, num_classes, input_shape):
#         K.clear_session() # Clear previous models from memory.
#         self.num_classes =  num_classes
#         self.input_shape = input_shape
#         self.model = ChimCLR()
#     def build(self):
#         model = keras.models.Sequential()
#         model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape))
#         model.add(keras.layers.MaxPooling2D((2, 2)))
#         model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
#         model.add(keras.layers.MaxPooling2D((2, 2)))
#         model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
#         model.add(keras.layers.Flatten())
#         model.add(keras.layers.Dense(64, activation='relu'))
#         model.add(keras.layers.Dense(self.num_classes, activation='softmax'))
        
#         model.compile(optimizer=tf.keras.optimizers.RMSprop(config.hyperparams['LEARNING_RATE']),
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics= ['sparse_categorical_accuracy'])
#         return model

class ChimCLR(tf.keras.Model):
    def __init__(self):
        super(ChimCLR, self).__init__()
        self.conv1 = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation=tf.nn.relu)
        self.pool1 = keras.layers.MaxPool2D(padding='SAME')
        self.conv2 = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation=tf.nn.relu)
        self.pool2 = keras.layers.MaxPool2D(padding='SAME')
        self.conv3 = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation=tf.nn.relu)
        self.pool3 = keras.layers.MaxPool2D(padding='SAME')
        self.pool3_flat = keras.layers.Flatten()
        self.dense4 = keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.drop4 = keras.layers.Dropout(rate=0.4)
        self.dense5 = keras.layers.Dense(units=10, activation=tf.nn.softmax)
    def call(self, inputs, training=False):
        net = self.conv1(inputs)
        net = self.pool1(net)
        net = self.conv2(net)
        net = self.pool2(net)
        net = self.conv3(net)
        net = self.pool3(net)
        net = self.pool3_flat(net)
        net = self.dense4(net)
        net = self.drop4(net)
        net = self.dense5(net)
        return net