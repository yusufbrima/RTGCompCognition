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
from sklearn.decomposition import  PCA,TruncatedSVD,FastICA
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from IPython.display import HTML
import matplotlib.animation as animation


plt.rcParams['text.usetex'] = False
plt.rcParams['font.size'] = 12


class Preprocessing:
    def __init__(self,filepath):
        self.INPUT_PATH  = filepath
        self.CLASSES =  self.get_classes()
    

    def get_classes(self):
        C = []
        for d in os.listdir(self.INPUT_PATH):
            temp =  Path(self.INPUT_PATH, d)
            if(os.path.isdir(temp) and len(glob.glob(f"{temp}/*.wav")) > 0):
                C.append(d)
        return C
    def get_files(self):

        ds =  {'file': [], 'class': [], 'label':  [], 'duration': [],'sr': []}
        for d in os.listdir(self.INPUT_PATH):
            temp =  Path(self.INPUT_PATH, d)
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
    
    def build_dataset(self,dur=config.audio['DURATION'], keepdims=True, crop_dims= (128,128),outpath=config.data['file_path']):
        X = []
        label = []
        Z = []
        CLASSES =  self.CLASSES
        for d in os.listdir(self.INPUT_PATH):
            temp =  Path(self.INPUT_PATH, d)
            if(os.path.isdir(temp) and len(os.listdir(temp)) > 0):
                for file in temp.glob("**/*.wav"):
                    filename =  Path(temp,file)
                    y,sr = librosa.load(filename, sr=None)
                    for i in range(0, len(y), sr*dur):
                        x = y[i:i+ (sr*dur)]
                        if(x.shape[0] < sr*dur):
                            z = np.zeros( abs(x.shape[0] -  (sr*dur)) )
                            x =  np.concatenate([x, z ])
                        D = librosa.stft(x,hop_length=config.audio['HOP_LENGTH'],n_fft= config.audio['FRAME_LENGHT']//4)  # STFT of y
                        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
                        Z.append(x)
                        if( not keepdims):
                            X.append(cv2.resize(S_db, crop_dims, interpolation = cv2.INTER_AREA))
                        else:
                            X.append(S_db)
                        label.append(CLASSES.index(d))
        print(f"{len(X)}  {dur} second audio samples created successfully")
        np.savez(outpath,x =  np.array(X), y = np.array(label), z= np.array(Z), c = CLASSES)




class Visualize:
    def __init__(self,figures,figsize=(24,8),dpi=300):
        self.figpath = figures
        self.figsize = figsize 
        self.dpi = dpi

    def show_distribution(self, df,save=True,filename='Data_Distribution_Default',figsize=(8,6)):
        plt.figure(1,figsize=figsize)
        ax = df['label'].value_counts().plot.barh(rot=0)
        plt.annotate(fr"$n = {df.shape[0]}$",(17.2,8.9),weight='bold',fontsize=12)
        ax.bar_label(ax.containers[0])
        plt.xlabel("Classes")
        plt.ylabel("Frequency")
        plt.tight_layout()
        if(save):
            plt.savefig(f'{self.figpath}/{filename}.png', bbox_inches ="tight", dpi=300)
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
                img = librosa.display.specshow(X[idx[i]],sr=config.audio['SAMPLE_RATE'],y_axis='mel', x_axis='time', hop_length=config.audio['HOP_LENGTH']//2, n_fft= config.audio['FRAME_LENGHT']) #
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