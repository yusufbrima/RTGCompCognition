import numpy as np
import librosa
import librosa.display
import soundfile as sf
import pandas as pd
import os
from pathlib import *
import glob
import shutil



DATA_PATH = Path('/net/store/cv/users/ybrima/scratch/data')
ZIP_PATH =  Path( DATA_PATH, os.listdir(DATA_PATH)[-1])
INPUT_PATH = Path('/net/store/cv/users/ybrima/scratch/data/archive/16000_pcm_speeches/')

CLASSES = os.listdir(INPUT_PATH)
FRAME_LENGHT =  1024
SAMPLE_RATE =  16000
HOP_LENGTH =  512
n_fft=2048
num_mfcc=13


def get_files(INPUT_PATH):
    ds =  {'file': [], 'class': []}
    for d in os.listdir(INPUT_PATH):
        temp =  Path(INPUT_PATH, d)
        if(os.path.isdir(temp)):
            for file in temp.glob("**/*.wav"):
                filename =  Path(temp,file)
                ds['file'].append(filename)
                ds['class'].append(CLASSES.index(d))
    data = pd.DataFrame(ds)
    print(f"{data.shape[0]} files read successfully")
    return data,CLASSES

def save_array(df,file_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    X  = []
    Q = []
    y =  []
    for k in range(df.shape[0]):
        r =  df.iloc[k]
        #   Loading the audio file 
        signal,sample_rate = librosa.load(r['file'], sr=SAMPLE_RATE)
        
        X.append(signal)
        y.append(y)
        
        # extract mfcc
        mfcc = librosa.feature.mfcc(signal, SAMPLE_RATE, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc = mfcc.T
        Q.append(mfcc)
    
    X =  np.array(X)
    Q =  np.array(Q)
    y =  np.array(y)
    np.savez(file_path,x=X,q=Q,y=y)
    print(f"Data written to storage successfully, path = {file_path}/speakers.npz")


if __name__ == '__main__':
    
    ds, CLASSES = get_files(INPUT_PATH)
    save_array(ds,INPUT_PATH)