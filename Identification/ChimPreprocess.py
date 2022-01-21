import tensorflow.keras.backend as K
import tensorflow.keras as keras
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
import numpy as np
import scipy as sp
import librosa
import librosa.display
import soundfile as sf
import pandas as pd
import os
from pathlib import *
import glob
import shutil


INPUT_PATH = Path('/net/store/cbc/projects/Pan troglodytes/audio_PH_dataset/PH/good_data/')
file_path =  '/net/store/cv/users/ybrima/scratch/data/Luango_Speaker/clip_loango.npz'
BASE_PATH =  '/net/store/cv/users/ybrima/scratch/data/Luango_Speaker/'
CLASSES = os.listdir(INPUT_PATH)
FRAME_LENGHT =  2048
SAMPLE_RATE =  44100
HOP_LENGTH =  512


def get_files(INPUT_PATH):
    ds =  {'file': [], 'class': [], 'label':  [], 'duration': []}
    for d in os.listdir(INPUT_PATH):
        temp =  Path(INPUT_PATH, d)
        if(os.path.isdir(temp)):
            for file in temp.glob("**/*.wav"):
                filename =  Path(temp,file)
                y,_ = librosa.load(filename, sr=SAMPLE_RATE)
                TRACK_DURATION = (1/SAMPLE_RATE)* len(y) # measured in seconds
                ds['file'].append(filename)
                ds['class'].append(CLASSES.index(d))
                ds['label'].append(d)
                ds['duration'].append(TRACK_DURATION)
    data = pd.DataFrame(ds)
    data.to_csv(Path(BASE_PATH,'metadata.csv'), index=False)
    print(f"{data.shape[0]} files read successfully")
    return data


def process_files(df):
    X  = []
    y =  []
    num_segments = 16
    num_mfcc= 32
    n_fft=1024
    dur = []
    for k in range(df.shape[0]):
        r =  df.iloc[k]

        #   Loading the audio file 
        yy,_ = librosa.load(r['file'], sr=SAMPLE_RATE)
        TRACK_DURATION = (1/SAMPLE_RATE)* len(yy) # measured in seconds
        SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
        samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)

        for i in range(0,len(yy),HOP_LENGTH):
            signal = yy[i:i+samples_per_segment]
            if(len(signal) == samples_per_segment):
                # extract mfcc
                D = librosa.amplitude_to_db(np.abs(librosa.stft(signal,hop_length=HOP_LENGTH)), ref=np.max)
                X.append(D)
                # print(D.shape)
                y.append(CLASSES.index(r['label']))
            break
        print(f"duration {TRACK_DURATION} file {Path(df.iloc[k,0]).name}")
        dur.append(TRACK_DURATION)
    # X =  np.array(X)
    # y =  np.array(y)
    # np.savez(file_path,x=X,y=y)
    print(f"Data written to storage successfully, path = {file_path}")
    print(max(dur), df.iloc[dur.index(max(dur)),0] )
if __name__ == '__main__':
    df = get_files(INPUT_PATH)
    process_files(df)
    print('Extraction completed successfully')
    
    
    