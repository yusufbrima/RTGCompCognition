import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import librosa
from pathlib import Path
import librosa.display
import matplotlib.pyplot as plt
import config
from utils import Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import  PCA,TruncatedSVD,FastICA

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 50




class DataLoader:
    def __init__(self,datapath, pt = .1,dur=config.audio['DURATION'],make=False,keepdims=True, crop_dims= (128,128)):
        K.clear_session() # Clear previous models from memory.
        self.datapath =  datapath
        self.pt = pt
        self.dur =  dur
        pp = Preprocessing(config.data['INPUT_PATH'])
        if Path(datapath).is_file() and not make:
            pass
        else:
            if(keepdims):
                pp.build_dataset( keepdims=True,dur=self.dur, outpath=datapath)
            else:
                pp.build_dataset( keepdims=keepdims,dur=self.dur, crop_dims= crop_dims,outpath=datapath)
        
    def load(self, flag = False):
        with np.load(self.datapath ,allow_pickle=True) as data:
            self.X = data['x']
            self.y = data['y']
            self.Z =  data['z']
            self.X_ = (self.X - self.X.mean(axis=0, keepdims=True))/self.X.std(axis=0,keepdims=True)
            if(flag):
                self.P = data['p']
                self.Q  = data['q']
                self.P_ = (self.P - self.P.mean(axis=0, keepdims=True))/self.P.std(axis=0,keepdims=True)
                self.Q_ = (self.Q - self.Q.mean(axis=0, keepdims=True))/self.Q.std(axis=0,keepdims=True)
            self.CLASSES = list(data['c'])
    def standardize(self, flag = False):
        self.X =  (self.X - self.X.mean(axis=0, keepdims=True))/self.X.std(axis=0,keepdims=True)
        if(flag):
            self.P = (self.P - self.P.mean(axis=0, keepdims=True))/self.P.std(axis=0,keepdims=True)
            self.Q = (self.Q - self.Q.mean(axis=0, keepdims=True))/self.Q.std(axis=0,keepdims=True)
    def project3D(self, Model, n_components = 3):
        model =  Model(n_components=n_components, random_state=np.random.seed(42))
        self.X_projection = model.fit_transform(self.X_.reshape(self.X_.shape[0],-1))
        if(len(str(model)) > 5):
            modelName = str(model)[:3]
        else:
            modelName = str(model)[:-2]
        print(f" Successfully projected {self.X.shape[1]} to {n_components} dimensions using {modelName}")
    
    def create_tensor_set(self, n = 0):
        if(n == 1):
            self.load(True)
            self.standardize(True)
            self.input_shape =  (self.P.shape[1],self.P.shape[2],1)
            self.output_nums =  len(self.CLASSES)
            self.P = self.P[...,np.newaxis]
            self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.P,self.y, test_size=self.pt, shuffle=True)
            self.X_train,self.X_valid,self.y_train,self.y_valid = train_test_split(self.X_train,self.y_train, test_size=self.pt, shuffle=True)
        elif(n ==  2):
            self.load(True)
            self.standardize(True)
            self.input_shape =  (self.Q.shape[1],self.Q.shape[2],1)
            self.output_nums =  len(self.CLASSES)
            self.Q = self.Q[...,np.newaxis]
            self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.Q,self.y, test_size=self.pt, shuffle=True)
            self.X_train,self.X_valid,self.y_train,self.y_valid = train_test_split(self.X_train,self.y_train, test_size=self.pt, shuffle=True)
        else:
            self.load()
            self.standardize()
            self.input_shape =  (self.X.shape[1],self.X.shape[2],1)
            self.output_nums =  len(self.CLASSES)
            self.X = self.X[...,np.newaxis]
            self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.y, test_size=self.pt, shuffle=True)
            self.X_train,self.X_valid,self.y_train,self.y_valid = train_test_split(self.X_train,self.y_train, test_size=self.pt, shuffle=True)

        train_dataset = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((self.X_test, self.y_test))
        valid_dataset = tf.data.Dataset.from_tensor_slices((self.X_valid, self.y_valid))
        self.train_dataset = train_dataset.shuffle(config.data['SHUFFLE_BUFFER_SIZE']).batch(config.data['BATCH_SIZE'])
        self.valid_dataset = valid_dataset.shuffle(config.data['SHUFFLE_BUFFER_SIZE']).batch(config.data['BATCH_SIZE']) #valid_dataset.batch(config.data['BATCH_SIZE'])
        self.test_dataset =  test_dataset.shuffle(config.data['SHUFFLE_BUFFER_SIZE']).batch(config.data['BATCH_SIZE'])  #test_dataset.batch(config.data['BATCH_SIZE'])



