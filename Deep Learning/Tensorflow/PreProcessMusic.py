import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd
import random
import json
import os
import math


BASE_DIR = f"/net/projects/scratch/winter/valid_until_31_July_2022/ybrima/data/genres/"


file_path = "/net/projects/scratch/winter/valid_until_31_July_2022/ybrima/data/data_genre.npz"
SAMPLE_RATE = 44100
n_fft = 2048
hop_length =  512
num_mfcc=32
TRACK_DURATION = 30 # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

def get_files(BASE_DIR):
    CLASSES = os.listdir(BASE_DIR)
    ds =  {'file':[], "class":[]}
    for c in CLASSES:
        if(os.path.isdir(os.path.join(BASE_DIR,c))):
            files =  os.listdir(os.path.join(BASE_DIR,c))
            cur_dir = os.path.join(BASE_DIR,c)
            for f in files:
                file =  os.path.join(cur_dir,f)
                ds['file'].append(file)
                ds['class'].append(c)
    data = pd.DataFrame(ds)
    return data,CLASSES




def save_array(df,file_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    X  = []
    y =  []
    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)
    for k in range(df.shape[0]):
        r =  df.iloc[k]

        #   Loading the audio file 
        signal,sample_rate = librosa.load(r['file'], sr=SAMPLE_RATE)
        
        # process all segments of audio file
        for d in range(num_segments):
            # calculate start and finish sample for current segment
            start = samples_per_segment * d
            finish = start + samples_per_segment

            # extract mfcc
            mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
            mfcc = mfcc.T
            # store only mfcc feature with expected number of vectors
            if len(mfcc) == num_mfcc_vectors_per_segment:
                X.append(mfcc)
                y.append(classes.index(r['class']))
                print("{}, segment:{}".format(r['file'], d+1))

    X =  np.array(X)
    y =  np.array(y)
    np.savez(file_path,x=X,y=y)
    print(f"Data written to storage successfully, path = {file_path}")


# def plot_sample(x,sr):
    
#     if(x.shape[0] > x.shape[1]):
#         x =  x.T
#     librosa.display.specshow(x, sr=sr, x_axis='time',y_axis="log",hop_length=hop_length)
#     plt.colorbar(format='%+2.f') 

if __name__ == "__main__":
    df,classes = get_files(BASE_DIR)
    save_array(df,file_path,num_mfcc=num_mfcc)