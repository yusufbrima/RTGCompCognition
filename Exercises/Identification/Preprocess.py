import numpy as np
import librosa
import librosa.display
import soundfile as sf
import pandas as pd
import os
from pathlib import *
import matplotlib.pyplot as plt
import glob

INPUT_PATH = Path('/net/store/cbc/projects/Pan troglodytes/audio_PH_dataset/PH/good_data/')
file_path =  '/net/store/cv/users/ybrima/scratch/data/Luango_Speaker/clip_loango.npz'
BASE_PATH =  '/net/store/cv/users/ybrima/scratch/data/Luango_Speaker/'
CLEAN_PATH =  Path(BASE_PATH,'good_data')
CLASSES = os.listdir(INPUT_PATH)
FRAME_LENGHT =  1024
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


def init_outfiles():
    CLEAN_PATH =  Path(BASE_PATH,'good_data')
    if(not os.path.exists(CLEAN_PATH)):
        os.mkdir(CLEAN_PATH)
        print(f'Base directory successfully created at {CLEAN_PATH}')
    for d in os.listdir(INPUT_PATH):
        if(not os.path.exists(Path(CLEAN_PATH,d))):
            os.mkdir(Path(CLEAN_PATH,d))
            print(f'Base directory successfully created at {Path(CLEAN_PATH,d)}')
    print(f'Process completed successfully {len(os.listdir(INPUT_PATH))} directories created')

def get_files_updated(INPUT_PATH):
    init_outfiles()
    df = get_files(INPUT_PATH)
    max_idx = df[df['duration'] == df['duration'].max()].index
    # min_idx = df[df['duration'] == df['duration'].min()].index
    yy,sr =  librosa.load(df.iloc[max_idx[0],0],sr=None)


    ds =  {'file': [], 'class': [], 'label':  [], 'duration': []}
    for d in os.listdir(INPUT_PATH):
        temp =  Path(INPUT_PATH, d)
        out_temp = Path(CLEAN_PATH, d)
        if(os.path.isdir(temp)):
            for file in temp.glob("**/*.wav"):
                filename =  Path(temp,file)
                outfile = Path(out_temp,file.name)
                y,sr = librosa.load(filename, sr=SAMPLE_RATE)
                dff =  int(np.ceil( len(yy) - len(y) ) / 2)
                a  = np.lib.pad(y, (int(np.floor(dff)) ,), 'symmetric')
                TRACK_DURATION = (1/SAMPLE_RATE)* len(a) # measured in seconds
                sf.write(outfile,a,sr)
                ds['file'].append(outfile)
                ds['class'].append(CLASSES.index(d))
                ds['label'].append(d)
                ds['duration'].append(TRACK_DURATION)
    data = pd.DataFrame(ds)
    data.to_csv(Path(BASE_PATH,'metadata_clean.csv'), index=False)
    print(f"{data.shape[0]} files preprocessed successfully")
    return data


def read_data():
    ds =  {'X': [], 'class': [], 'label':  [],'spec': []}

    df = pd.read_csv(Path(BASE_PATH,'metadata_clean.csv'))
    for k in range(df.shape[0]):
        r =  df.iloc[k]
        #   Loading the audio file 
        y,_ = librosa.load(r['file'], sr=None)


        D = librosa.amplitude_to_db(np.abs(librosa.stft(y,n_fft=2048, hop_length=HOP_LENGTH)), ref=np.max)

        ds['X'].append(y)
        ds['spec'].append(D)
        ds['class'].append(r['class'])
        ds['label'].append(r['label'])
    dff =  pd.DataFrame(ds)
    dff.to_csv(Path(BASE_PATH,'clean_data.csv'), index=False)
    print(f"{dff.shape[0]} waveforms extracted successfully ")

if __name__ == '__main__':
    """
        This code allows us to preprocess the dataset using varied padding techniques to make the input sequence into fixed length
     """
    df = get_files(INPUT_PATH)

    df = get_files_updated(INPUT_PATH)

    read_data()