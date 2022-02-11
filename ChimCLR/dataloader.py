import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import config
from utils import Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import  PCA,TruncatedSVD,FastICA

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 50




class DataLoader:
    def __init__(self,datapath, pt = .1,keepdims=False, crop_dims= (128,128)):
        K.clear_session() # Clear previous models from memory.
        self.datapath =  datapath
        self.pt = pt
        pp = Preprocessing(config.data['INPUT_PATH'])
        if(keepdims):
            pp.build_dataset( keepdims=True,outpath=datapath)
        else:
            pp.build_dataset( keepdims=keepdims, crop_dims= crop_dims,outpath=datapath)
        
    def load(self):
        with np.load(self.datapath ,allow_pickle=True) as data:
            self.X = data['x']
            self.y = data['y']
            self.Z =  data['z']
            self.X_ = (self.X - self.X.mean(axis=0, keepdims=True))/self.X.std(axis=0,keepdims=True)
            self.CLASSES = list(data['c'])
    def standardize(self):
        self.X =  (self.X - self.X.mean(axis=0, keepdims=True))/self.X.std(axis=0,keepdims=True)
    def project3D(self, Model, n_components = 3):
        model =  Model(n_components=n_components, random_state=np.random.seed(42))
        self.X_projection = model.fit_transform(self.X_.reshape(self.X_.shape[0],-1))
        if(len(str(model)) > 5):
            modelName = str(model)[:3]
        else:
            modelName = str(model)[:-2]
        print(f" Successfully projected {self.X.shape[1]} to {n_components} dimensions using {modelName}")
    
    def create_tensor_set(self):
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
        self.valid_dataset = valid_dataset.batch(config.data['BATCH_SIZE'])
        self.test_dataset = test_dataset.batch(config.data['BATCH_SIZE'])



