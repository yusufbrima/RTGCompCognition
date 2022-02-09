from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from utils import Visualize
from dataloader import DataLoader
import tensorflow as tf
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from tensorflow.keras.applications import VGG16,VGG19, ResNet50,InceptionV3,Xception
from model import ChimCLR,make_model, autoencoder
from train import train_model
import config


tf.random.set_seed(42)

MODELS = [VGG16,VGG19, ResNet50,InceptionV3,Xception]

if __name__ == "__main__":
     viz =  Visualize(config.figures['figpath'])
     dl =  DataLoader(datapath = config.data['file_path2'])
     dl.load()
     X =  dl.X 
     X =  (X - X.mean(axis=0, keepdims=True))/X.std(axis=0,keepdims=True)
     input_shape =  (X.shape[1],X.shape[2],1)
     output =  len(dl.CLASSES)
     X = X[...,np.newaxis]
     X_train,X_test,y_train,y_test = train_test_split(X,dl.y, test_size=0.1, shuffle=True)

     encoder, decoder,conv_autoencoder = autoencoder(input_shape)
     history = conv_autoencoder.fit(X_train, X_train, batch_size=64, epochs=40, validation_data=(X_test, X_test))



     sample_idx = np.random.randint(0,X_train.shape[0], 4)
     samples = X_train[sample_idx]
     latent =  encoder(samples).numpy()
     decoded  =  decoder(latent).numpy() 
     fig = plt.figure(1, figsize = (20,12))
     locs = [4,5,6,7]
     j  = 1
     for i, (s, d) in enumerate(zip(samples, decoded)):
          ax = plt.subplot(4,4,i+1)
          librosa.display.specshow(s.reshape(128, 128), sr=config.audio['SAMPLE_RATE'],x_axis='time', y_axis='mel', hop_length=config.audio['HOP_LENGTH'])
          plt.title("Ground Truth")
          ax = plt.subplot(4,4,i+4)
          librosa.display.specshow(d.reshape(128, 128), sr=config.audio['SAMPLE_RATE'],x_axis='time', y_axis='mel',hop_length=config.audio['HOP_LENGTH'])
          plt.title("Reconstructed")
          if(i == 3):
               ax = plt.subplot(4,4,i+5)
               librosa.display.specshow(d.reshape(128, 128), sr=config.audio['SAMPLE_RATE'], x_axis='time', y_axis='mel',hop_length=config.audio['HOP_LENGTH'])
               plt.title("Reconstructed")
     plt.tight_layout()
     plt.savefig(f'{config.figures["figpath"]}/Reconstruction_Chart.png', bbox_inches ="tight", dpi=300)
     plt.show()
     # dl.create_tensor_set()
     # df,dft = train_model(MODELS,dl,dl.input_shape,len(dl.CLASSES))

     # print(df.head())
     # df.to_csv('./Data/model_results_train.csv', index = False)
     # dft.to_csv('./Data/model_results_test.csv', index = False)

