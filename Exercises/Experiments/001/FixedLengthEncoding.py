import librosa
from pathlib import Path
import numpy as np
import pandas as pd
import random
import os
import math

BASE_DIR = "/net/projects/scratch/winter/valid_until_31_July_2022/krumnack/animal-communication-data/Chimp_IvoryCoast/manually_verified_2s/chimp_only_23112020_with_ids"
file_path = "/net/projects/scratch/winter/valid_until_31_July_2022/ybrima/data/data_chimp_preprocessed.npz"
selected_path = "/net/projects/scratch/winter/valid_until_31_July_2022/ybrima/data/data_chimp.csv"
CLASSES = ['kub','woo','rom','uta','jac','ish','sum','jul','kin','Kuba']
SAMPLE_RATE = 44100
N_FFT = 2048
HOP_LENGTH =  512
num_mfcc=32

def sanitize(INPUT_PATH):
    ds =  {'file': [],'class': [],'duration':[], 'sr':[]}
    for f in Path(INPUT_PATH).glob("**/*.wav"):
        ds['file'].append(f)
        f =  str(f)
        y,sr =  librosa.load(f, sr=None)
        ds['duration'].append((1/sr) * len(y))
        ds['sr'].append(sr)
        clx = f[f.rfind("/")+12:][:f[f.rfind("/")+12:].index('_')]
        if(len(clx) > 3):
            ds['class'].append(clx.split('-')[0])
        else:
            ds['class'].append(clx)
    data =  pd.DataFrame(ds)
    df  = data[data['class'].map(data['class'].value_counts()) > 150]
    return df

def preprocess():
    ds =  {'data': [],'class': [], 'sr': [], "spec" : []}
    df =  pd.read_csv(selected_path)
    max_idx = df[df['duration'] == df['duration'].max()].index
    if(len(max_idx) > 1):
        max_idx =  max_idx[0]
    r = df.iloc[max_idx]
    filename = r['file']
    ymax,sr = librosa.load(filename, sr=r['sr'])
    for k in range(df.shape[0]):
        r = df.iloc[k]
        y,sr = librosa.load(r['file'], sr=r['sr'])
        dff =  int(np.ceil( len(ymax) - len(y) ) / 2)
        a  = np.lib.pad(y, (int(np.floor(dff)) ,), 'symmetric')
        if(len(a)  ==  len(ymax)):
            ds['data'].append(a)
            ds['class'].append(CLASSES.index(r['class']))
            ds['sr'].append(sr)
            D = librosa.amplitude_to_db(np.abs(librosa.stft(a,n_fft=2048, hop_length=HOP_LENGTH)), ref=np.max)
            ds['spec'].append(D)
    X =  np.array(ds['data'])
    y = np.array(ds['class'])
    Z  =  np.array(ds['spec'])
    sr = np.array(ds['sr'])
    np.savez(file_path,X=X,y=y,sr=sr,Z = Z)
    del ds 
    print(f"Data preprocessed successfully and saved to {file_path}")

if __name__ == '__main__':
    """ This script implements the fixed length encoding module """
    df =  sanitize(BASE_DIR)
    df.to_csv(selected_path,index=False)
    preprocess()


