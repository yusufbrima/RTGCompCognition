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

# INPUT_PATH = Path('/net/store/cbc/projects/Pan troglodytes/audio_PH_dataset/PH/good_data/')
file_path =  '/net/store/cv/users/ybrima/scratch/data/Luango_Speaker/clip_loango.npz'


FRAME_LENGHT =  1024
SAMPLE_RATE =  44100
HOP_LENGTH =  512

def get_classes(INPUT_PATH):
    C = []
    for d in os.listdir(INPUT_PATH):
        temp =  Path(INPUT_PATH, d)
        if(os.path.isdir(temp) and len(glob.glob(f"{temp}/*.wav")) > 0):
            C.append(d)
    return C

def build_dataset(INPUT_PATH, dur=1, keepdims=True, crop_dims= (128,128),outpath=file_path):
    X = []
    label = []
    Z = []
    CLASSES =  get_classes(INPUT_PATH)
    for d in os.listdir(INPUT_PATH):
        temp =  Path(INPUT_PATH, d)
        if(os.path.isdir(temp) and len(os.listdir(temp)) > 0):
            for file in temp.glob("**/*.wav"):
                filename =  Path(temp,file)
                y,sr = librosa.load(filename, sr=None)
                for i in range(0, len(y), sr*dur):
                  x = y[i:i+ (sr*dur)]
                  if(x.shape[0] < sr*dur):
                    z = np.zeros( abs(x.shape[0] -  (sr*dur)) )
                    x =  np.concatenate([x, z ])
                  D = librosa.stft(x,hop_length=HOP_LENGTH,n_fft= FRAME_LENGHT//4)  # STFT of y
                  S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
                  Z.append(x)
                  if( not keepdims):
                      X.append(cv2.resize(S_db, crop_dims, interpolation = cv2.INTER_AREA))
                  else:
                      X.append(S_db)
                  label.append(CLASSES.index(d))
    print(f"{len(X)}  {dur} second audio samples created successfully")
    np.savez(outpath,x =  np.array(X), y = np.array(label), z= np.array(Z), c = CLASSES)
    return np.array(X), np.array(label), np.array(Z)


if __name__ == '__main__':
    """
        This code allows us to preprocess the dataset using varied padding techniques to make the input sequence into fixed length
     """
    pass