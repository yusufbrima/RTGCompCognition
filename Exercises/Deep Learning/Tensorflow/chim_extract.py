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


BASE_DIR = "/net/projects/scratch/winter/valid_until_31_July_2022/krumnack/animal-communication-data/Chimp_IvoryCoast/manually_verified_2s/chimp_only_23112020_with_ids"
file_path = "/net/projects/scratch/winter/valid_until_31_July_2022/ybrima/data/data_chimp.npz"
SAMPLE_RATE = 44100
n_fft = 2048
hop_length =  512
num_mfcc=32
def get_files(path):
    files = [os.path.join(path,f) for f in os.listdir(path) if os.path.isfile(os.path.join(path,f)) and f.endswith('.wav')]
    return files


def sanitize(files):
    ds =  {'file':[], "class":[]}

    for f in files:

        ds['file'].append(f)
        ds['class'].append(f[f.rfind("/")+12:][:f[f.rfind("/")+12:].index('_')])
        
    #next we want to select files that belong to class >=  100 samples
    
    # pd.Series(data['class'].unique()).sort_values()
    l = ['kub','woo','rom','jac','sum','kub-phsm','kub-phtbsm','uta','ish-phsm','jul','rom-phsm','kub-phtb','woo-phsm']
    data = pd.DataFrame(ds)
    
    #next we concatinate the selected samples per class into a single data frame
    
    df = data.loc[data['class'] == l[0]].copy()
    for i in range(1,len(l)):
        df = pd.concat([df, data.loc[data['class'] == l[i]]], axis = 0)
    return df,l




def save_array(df,file_path, num_mfcc=13, n_fft=2048, hop_length=512):
    X  = []
    y =  []
    for k in range(df.shape[0]):
        r =  df.iloc[k]

        #   Loading the audio file 
        signal,sample_rate = librosa.load(r['file'], sr=SAMPLE_RATE)
        
        # extract mfcc
        mfcc = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc = mfcc.T
        w,h = 173, num_mfcc
        # print(mfcc.shape)
        # break
        # store only mfcc feature with expected number of vectors
        if(mfcc.shape[0] == w and mfcc.shape[1] == h):
            X.append(mfcc)
            y.append(classes.index(r['class']))
            print("{}".format(r['file']))

    X =  np.array(X)
    y =  np.array(y)
    np.savez(file_path,x=X,y=y)
    print(f"Data written to storage successfully, path = {file_path}")


if __name__ == "__main__":
    files =  get_files(BASE_DIR)
    df,classes = sanitize(files)
    save_array(df,file_path,num_mfcc=num_mfcc)