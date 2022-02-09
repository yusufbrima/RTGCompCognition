from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.decomposition import  PCA,TruncatedSVD,FastICA
from utils import Visualize
from dataloader import DataLoader
import config
from utils import Preprocessing,Animation

plt.rcParams['text.usetex'] = False
plt.rcParams['font.size'] = 12

tf.random.set_seed(42)

if __name__ == '__main__':
    """
        This code allows us to preprocess the dataset using varied padding techniques to make the input sequence into fixed length
     """

    viz =  Visualize(config.figures['figpath'])


    # dl.load() #This function reads the compressed numpy arrays into X,y,Z for the spectogram, labels, and 1 second wave forms respectively

    pp =  Preprocessing(config.data['INPUT_PATH'])
    # viz.display(dl.X,dl.Z,dl.y, dl.CLASSES, save=False, flag=True)


    dl = DataLoader(config.data['file_path2'],keepdims=False, crop_dims= (128,128) )
    dl.load()
    anim = Animation(PCA,dl.X,dl.y,dl.CLASSES, config.figures['figpath'])

    anim.project3D(n = 3, save=True, filename= 'PCA_Projection',frames=360, interval=20)