from __future__ import print_function
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import config
from pathlib import Path
import pandas as pd
import glob
import cv2
import os
import librosa
import scipy.signal
import soundfile as sp
from sklearn.decomposition import  PCA,TruncatedSVD,FastICA
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from IPython.display import HTML
import matplotlib.animation as animation
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
from inaSpeechSegmenter import Segmenter
from inaSpeechSegmenter.export_funcs import seg2csv, seg2textgrid
import warnings
from pydub import AudioSegment
from pydub.utils import mediainfo
from pydub.playback import play
import subprocess

warnings.simplefilter(action='ignore', category=FutureWarning)

plt.rcParams['text.usetex'] = False
plt.rcParams['font.size'] = 12


class Preprocessing:
    def __init__(self,filepath):
        self.INPUT_PATH  = filepath
        self.CLASSES =  self.get_classes()
    

    def get_classes(self):
        C = []
        for d in os.listdir(config.data['CONVERT_PATH']):
            temp =  Path(config.data['CONVERT_PATH'], d)
            if(os.path.isdir(temp) and len(glob.glob(f"{temp}/*.wav")) > 0):
                C.append(d)
        return C
    def segment_file(self,f):
        # create an instance of speech segmenter
        # this loads neural networks and may last few seconds
        # Warnings have no incidence on the results
        seg = Segmenter()
        # segmentation is performed using the __call__ method of the segmenter instance
        segmentation = seg(f)
        return segmentation
    
    def sanitize_file(self,f):
        segmentation =  self.segment_file(f)
        
        y,sr =  librosa.load(f, sr= None)
        duration  =  (1/sr) * len(y)
        offset = self.indexer(segmentation)
        if(offset is not None):
            if(float(offset) <= float(duration) and float(offset) != .0):
                y,sr =  librosa.load(f, duration=offset, sr= None)
        y =  self.applyfilter(y)
        return y,sr
    
    def indexer(self,segmentation):
        for i,s in enumerate(segmentation):
            if(s[0] == 'female' or s[0] == 'music'  and s[1] != 0.0 ):
                return s[1]

    def init_outfiles(self, path ):
        CLEAN_PATH =  path
        if(not os.path.exists(CLEAN_PATH)):
            os.mkdir(CLEAN_PATH)
            print(f'Base directory successfully created at {CLEAN_PATH}')
        for d in os.listdir(config.data['INPUT_PATH']):
            if(not os.path.exists(Path(CLEAN_PATH,d))):
                os.mkdir(Path(CLEAN_PATH,d))
                print(f'Base directory successfully created at {Path(CLEAN_PATH,d)}')
        print(f'Process completed successfully {len(os.listdir(config.data["INPUT_PATH"]))} directories created')
    
    def applyfilter(self,y, pt = 0.1):
        # apply a 3-pole lowpass filter at 0.1x Nyquist frequency
        b, a = scipy.signal.butter(3, pt)
        filtered = scipy.signal.filtfilt(b, a, y )
        return filtered

    def applyWindowing(self,y, windowSize=40):
        # create a normalized Hanning window
        window = np.hanning(windowSize)
        window = window / window.sum()

        # filter the data using convolution
        filtered = np.convolve(window, y, mode='valid')
        return filtered
    
    def clean(self):
        CLEAN_PATH =  config.data['OUT_PATH']
        self.init_outfiles(CLEAN_PATH)
        for d in os.listdir(config.data['CONVERT_PATH']):
            temp =  Path(config.data['CONVERT_PATH'], d)
            temp_out = Path(CLEAN_PATH,d)
            if(os.path.isdir(temp)):
                for file in temp.glob("**/*.wav"):
                    filename =  Path(temp,file)
                    print(f"We are cleaning {filename}")
                    y,sr =  self.sanitize_file(filename)
                    sp.write(str(Path(temp_out,file.name)),  y, sr)
        print("Dataset cleaning completed successfully. Process exited")
    
    def convert(self):
        CLEAN_PATH =  config.data['CONVERT_PATH']
        self.init_outfiles(CLEAN_PATH)
        for d in os.listdir(self.INPUT_PATH):
            temp =  Path(self.INPUT_PATH, d)
            temp_out = Path(CLEAN_PATH,d)
            if(os.path.isdir(temp)):
                for file in self.get_files_multiple(temp):
                    filename =  Path(temp,file)
                    if(filename.suffix == '.mp3'):
                        subprocess.call(['sox',str(filename), '-e', 'mu-law','-r', '44.1k',str(Path(temp_out, f'{filename.name.split(".")[0]}.wav')), 'remix', '1,2'])
                    else:
                        y, sr =  librosa.load(str(filename), sr= None)
                        sp.write(str(Path(temp_out, filename.name )), y, sr)
        print("Converted Audio Signal into wav format successfully")
    
    def get_files_multiple(self,path):
        all_files = []
        extensions  = ('*.mp3', '*.wav')
        for ext in extensions:
            all_files.extend(Path(str(path)).glob(ext))
        return all_files
    
    def get_files(self):

        ds =  {'file': [], 'class': [], 'label':  [], 'duration': [],'sr': []}
        for d in os.listdir(config.data['CONVERT_PATH'])self.:
            temp =  Path(config.data['CONVERT_PATH'], d)
            if(os.path.isdir(temp)):
                for file in temp.glob("**/*.wav"):
                    filename =  Path(temp,file)
                    y,sr = librosa.load(filename, sr=None)
                    duration =  (1/sr) * len(y)
                    ds['file'].append(filename)
                    ds['class'].append(self.CLASSES.index(d))
                    ds['label'].append(d)
                    ds['duration'].append(duration)
                    ds['sr'].append(sr)
        data = pd.DataFrame(ds)
        data.to_csv(Path(config.data['BASE_PATH'],'metadata.csv'), index=False)
        print(f"{data.shape[0]} files read successfully")
        return data
    def build_single_ds(self, dur=1, outpath = config.data['file_path_pan'], keepdims = True,crop_dims= (128,128)):
        df =  self.get_files()
        df = df[(df.label == 'PAN')].copy()
        X = [] #stores the computed db scaled power spectrum of 1 second audio segments
        Z = [] #stores waveforms of the 1 second segments
        Q = [] #stores the melspectrums of the audio signals
        P = [] #stores the mfccs of the audio signal
        for i in range(df.shape[0]):
            filename =  Path(df.iloc[i,0])
            y,sr = librosa.load(filename, sr=None)
            for i in range(0, len(y), sr*dur):
                x = y[i:i+ (sr*dur)]
                if(x.shape[0] < sr*dur):
                    z = np.zeros( abs(x.shape[0] -  (sr*dur)) )
                    x =  np.concatenate([x, z ])
                D = librosa.stft(x,hop_length=config.audio['HOP_LENGTH']//2,n_fft= config.audio['FRAME_LENGHT']//4)  
                S_db = librosa.power_to_db(np.abs(D), ref=np.max)
                S = librosa.feature.melspectrogram(y=x, sr=config.audio['SAMPLE_RATE'],hop_length=config.audio['HOP_LENGTH']//2,n_fft= config.audio['FRAME_LENGHT']//4, n_mels=128,fmax=8000)
                mfcc = librosa.feature.mfcc(y= x,sr = config.audio['SAMPLE_RATE'],hop_length=config.audio['HOP_LENGTH']//2,n_fft= config.audio['FRAME_LENGHT']//4, n_mfcc=40)
                Z.append(x)
                if( not keepdims):
                    X.append(cv2.resize(S_db, crop_dims, interpolation = cv2.INTER_AREA))
                    Q.append(cv2.resize(S, crop_dims, interpolation = cv2.INTER_AREA))
                    P.append(cv2.resize(mfcc, crop_dims, interpolation = cv2.INTER_AREA))
                else:
                    X.append(S_db)
                    Q.append(S)
                    P.append(mfcc)
        print(f"{len(X)}  {dur} second audio samples for Pandy created successfully")
        np.savez(outpath,x =  np.array(X), z= np.array(Z), p =  np.array(P), q = np.array(Q))
    def build_dataset(self,dur=config.audio['DURATION'], keepdims=True, crop_dims= (128,128),outpath=config.data['file_path']):
        X = [] #stores the computed db scaled power spectrum of 1 second audio segments
        label = []
        Z = [] #stores waveforms of the 1 second segments
        Q = [] #stores the melspectrums of the audio signals
        P = [] #stores the mfccs of the audio signal

        CLASSES =  self.CLASSES
        for d in os.listdir(config.data['CONVERT_PATH']):
            temp =  Path(config.data['CONVERT_PATH'], d)
            if(os.path.isdir(temp) and len(os.listdir(temp)) > 0):
                for file in temp.glob("**/*.wav"):
                    filename =  Path(temp,file)
                    y,sr = librosa.load(filename, sr=None)
                    for i in range(0, len(y), sr*dur):
                        x = y[i:i+ (sr*dur)]
                        if(x.shape[0] < sr*dur):
                            z = np.zeros( abs(x.shape[0] -  (sr*dur)) )
                            x =  np.concatenate([x, z ])
                        D = librosa.stft(x,hop_length=config.audio['HOP_LENGTH']//2,n_fft= config.audio['FRAME_LENGHT']//4)  # STFT of y
                        # S_full, phase = librosa.magphase(librosa.stft(x, hop_length=config.audio['HOP_LENGTH']))
                        S_db = librosa.power_to_db(np.abs(D), ref=np.max)
                        S = librosa.feature.melspectrogram(y=x, sr=config.audio['SAMPLE_RATE'],hop_length=config.audio['HOP_LENGTH']//2,n_fft= config.audio['FRAME_LENGHT']//4, n_mels=128,fmax=8000)
                        mfcc = librosa.feature.mfcc(y= x,sr = config.audio['SAMPLE_RATE'],hop_length=config.audio['HOP_LENGTH']//2,n_fft= config.audio['FRAME_LENGHT']//4, n_mfcc=40)
                        Z.append(x)
                        if( not keepdims):
                            X.append(cv2.resize(S_db, crop_dims, interpolation = cv2.INTER_AREA))
                            Q.append(cv2.resize(S, crop_dims, interpolation = cv2.INTER_AREA))
                            P.append(cv2.resize(mfcc, crop_dims, interpolation = cv2.INTER_AREA))
                        else:
                            X.append(S_db)
                            Q.append(S)
                            P.append(mfcc)
                        label.append(CLASSES.index(d))
        print(f"{len(X)}  {dur} second audio samples created successfully")
        np.savez(outpath,x =  np.array(X), y = np.array(label), z= np.array(Z), c = CLASSES, p =  np.array(P), q = np.array(Q))




class Visualize:
    def __init__(self,figures,figsize=(24,8),dpi=300):
        self.figpath = figures
        self.figsize = figsize 
        self.dpi = dpi
    def plot_confusion_matrix(self, y_true, y_pred,CLASSES, save=True, filename='confusion_matrix',figsize=(8,6)):
        fig = plt.figure(1,figsize=figsize)
        df_cm = pd.DataFrame(confusion_matrix(y_true, y_pred), index = [i for i in CLASSES],columns = [i for i in CLASSES]) 
        sns.heatmap(df_cm, annot=True, cmap = 'Blues')
        if(save):
            plt.savefig(f'{self.figpath}/{filename}_{config.experiment["NAME"]}.png', bbox_inches ="tight", dpi=300)
        plt.close(fig)
    def show_distribution(self, df,save=True,filename='Data_Distribution_Default',figsize=(8,6)):
        plt.figure(1,figsize=figsize)
        ax = df['label'].value_counts().plot.barh(rot=0)
        plt.annotate(fr"$n = {df.shape[0]}$",(90.2,9.4),weight='bold',fontsize=12)
        ax.bar_label(ax.containers[0])
        plt.ylabel("Classes")
        plt.xlabel("Frequency")
        plt.tight_layout()
        if(save):
            plt.savefig(f'{self.figpath}/{filename}_{config.experiment["NAME"]}.png', bbox_inches ="tight", dpi=300)
        plt.show()
    def plot_components(self,X_pca,y,CLASSES, x_str='Component 1', y_str='Component 2', str_title="PCA", title="title.pdf"):
        fig = plt.figure(1,figsize=(10,6))
        ax =  fig.add_subplot(111)
        scatter = ax.scatter(X_pca[:,0],X_pca[:,1], c=list(y))
        ax.set_xlabel(x_str)
        ax.set_ylabel(y_str)
        ax.set_title(str_title)
        # ax.set_zlabel("PCA Component 3")
        ax.legend(handles=scatter.legend_elements()[0], labels=CLASSES,bbox_to_anchor=(1.0, 1.0))
        plt.savefig(f'{self.figpath}/{title}', bbox_inches ="tight", dpi=100)
        plt.show()

    def display(self, X,Z,y,CLASSES, idx,flag =  False, save=True,filename='Sample_Plot.png'):
        fig = plt.figure(1,figsize=self.figsize)
        for i in range(len(idx)):
            ax = plt.subplot(2,3,i+1)
            if(flag ==  False):
                librosa.display.waveshow(Z[idx[i]],sr=config.audio['SAMPLE_RATE'],alpha=0.6)
                plt.xlabel("time")
                plt.ylabel("amplitude")
                plt.title(CLASSES[y[idx[i]]].replace("_", " "))
            else:
                img = librosa.display.specshow(X[idx[i]],sr=config.audio['SAMPLE_RATE'],y_axis='log', x_axis='time', hop_length=config.audio['HOP_LENGTH']//2) #
                fig.colorbar(img, ax=ax, format='%+2.0f dB')
                plt.title(CLASSES[y[idx[i]]].replace("_", " "))
        plt.tight_layout()
        if(flag ==  False):
            plt.savefig(f'{self.figpath}/{filename}', bbox_inches ="tight", dpi=self.dpi)
        else:
            if(save):
                plt.savefig(f'{self.figpath}/{filename}', bbox_inches ="tight", dpi=self.dpi)
        plt.show()



class Animation:
    anim =  None
    fig = plt.figure(3, figsize=(10,6))
    fig.patch.set_facecolor(None)
    fig.patch.set_alpha(0.0)
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.axis('off')


    def __init__(self, Model, X,y, CLASSES, figures, dpi = 300):
        self.Model =  Model 
        self.X = X
        self.y = y 
        self.CLASSES = CLASSES
        self.figpath =  figures
        self.dpi = dpi

    def play3D(self,n = 3, save=True, filename= 'Projection_Animation',frames=360, interval=20):
        model =  self.Model(n_components=n, random_state=np.random.seed(42))
        self.X_projection = model.fit_transform(self.X.reshape(self.X.shape[0],-1))
        self.n = len(self.X_projection[:,0])
        self.xx = self.X_projection[:,0]
        self.yy = self.X_projection[:,1]
        self.zz = self.X_projection[:,2]

        # Animate
        ani = animation.FuncAnimation(self.fig, self.animate, init_func=self.init,
                                    frames=360, interval=100,blit=True)
         
        
        HTML(ani.to_html5_video())

    # def init(self):
    #     self.scatter = self.ax.scatter(self.xx, self.yy, self.zz, marker='o', s=20, c=list(self.y), alpha=0.6)
    #     return self.fig,
        
    # def animate(self,i):
    #     self.ax.view_init(elev=30., azim=3.6*i)
    #     return self.fig,

    



    def project3D(self,n = 3, save=True, filename= 'Projection_Animation',frames=360, interval=20):
        model =  self.Model(n_components=n, random_state=np.random.seed(42))
        self.X_projection = model.fit_transform(self.X.reshape(self.X.shape[0],-1))
        self.n = len(self.X_projection[:,0])
        self.xx = self.X_projection[:,0]
        self.yy = self.X_projection[:,1]
        self.zz = self.X_projection[:,2]

        # Create a figure and a 3D Axes
        # self.fig = plt.figure(3, figsize=(10,6))
        # self.fig.patch.set_facecolor(None)
        # self.fig.patch.set_alpha(0.0)
        # self.ax =   Axes3D(self.fig) #self.fig.add_subplot(111, projection='3d')
        # self.ax.grid(False)
        # self.ax.xaxis.pane.fill = False
        # self.ax.yaxis.pane.fill = False
        # self.ax.zaxis.pane.fill = False
        # self.ax.axis('off')
        print("We are here in execution")
        self.anim = animation.FuncAnimation(self.fig, self.animate, init_func=self.init,
                                    frames=frames, interval=interval, blit=True)

        HTML(self.anim.to_html5_video())
        # if(save):
        #     print("Converting animation to mp4 format and saving ...")
        #     self.anim.save(f'{self.figpath}/{filename}.mp4',dpi=self.dpi,savefig_kwargs={'frameon': False,'pad_inches': 'tight'})
        #     print("Converting animation to gif format and saving ...")
        #     os.system("cd Figures")
        #     os.system(f"ffmpeg -i {filename}.mp4 -filter:v fps=fps=30 {filename}.gif")
        #     os.system('cd ..')
        #     print(f"Animation Converted and saved successfully to animation to {self.figpath}/{filename}.gif")

    # Create an init function and the animate functions.
    # Both are explained in the tutorial. Since we are changing
    # the the elevation and azimuth and no objects are really
    # changed on the plot we don't have to return anything from
    # the init and animate function. (return value is explained
    # in the tutorial.
    def init(self):
        self.scatter = self.ax.scatter(self.xx, self.yy, self.zz, marker='o', s=20, c=list(self.y), alpha=0.6)
        self.ax.set_xlabel("PCA 1")
        self.ax.set_ylabel("PCA 2")
        self.ax.set_zlabel("PCA 3")
        self.ax.legend(handles=self.scatter.legend_elements()[0], labels=self.CLASSES,bbox_to_anchor=(1.2, 0.8))
        return self.fig,

    def animate(self,i):
        self.ax.view_init(elev=10., azim=i)
        return self.fig,



if __name__ == '__main__':
    """
        This code allows us to preprocess the dataset using varied padding techniques to make the input sequence into fixed length
     """

    pass