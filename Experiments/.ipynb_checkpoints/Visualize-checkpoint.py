import numpy as np
import librosa
import librosa.display
import soundfile as sf
import pandas as pd
import os
from pathlib import *
import matplotlib.pyplot as plt
import glob
import cv2

plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 12


def plot_components(X_pca, x_str='Component 1', y_str='Component 2', str_title="PCA", title="title.pdf"):
  fig = plt.figure(1,figsize=(10,6))
  ax =  fig.add_subplot(111)
  scatter = ax.scatter(X_pca[:,0],X_pca[:,1], c=list(y))
  ax.set_xlabel(x_str)
  ax.set_ylabel(y_str)
  ax.set_title(str_title)
  # ax.set_zlabel("PCA Component 3")
  ax.legend(handles=scatter.legend_elements()[0], labels=CLASSES,bbox_to_anchor=(1.0, 1.0))
  plt.savefig(f'{figures}/{title}', bbox_inches ="tight", dpi=100)
  plt.show()
if __name__ == '__main__':
    """
        This code allows us to preprocess the dataset using varied padding techniques to make the input sequence into fixed length
     """
    pass