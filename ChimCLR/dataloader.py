import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import config
from sklearn.model_selection import train_test_split


BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 50




class DataLoader:
    def __init__(self, pt = .1, datapath = config.data['file_path']):
        K.clear_session() # Clear previous models from memory.
        self.datapath =  datapath
        self.pt = pt
    def load(self):
        with np.load(self.datapath ,allow_pickle=True) as data:
            self.X = data['x']
            self.y = data['y']
            self.Z =  data['z']
            self.CLASSES = list(data['c'])
    def standardize(self):
        self.X =  (self.X - self.X.mean(axis=0, keepdims=True))/self.X.std(axis=0,keepdims=True)
    
    def create_tensor_set(self):
        self.load()
        self.standardize()
        self.input_shape =  (self.X.shape[1],self.X.shape[2],1)
        self.output_nums =  len(self.CLASSES)
        self.X = self.X[...,np.newaxis]

        self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.y, test_size=self.pt, shuffle=True)

        train_dataset = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((self.X_test, self.y_test))
        self.train_dataset = train_dataset.shuffle(config.data['SHUFFLE_BUFFER_SIZE']).batch(config.data['BATCH_SIZE'])
        self.test_dataset = test_dataset.batch(config.data['BATCH_SIZE'])



