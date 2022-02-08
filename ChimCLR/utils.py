import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import config

plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 12


class Visualize:
    def __init__(self,figures,figsize=(24,8),dpi=300):
        self.figpath = figures
        self.figsize = figsize 
        self.dpi = dpi

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

    def display(self, X,Z,y,CLASSES,n=6,flag =  False, save=True,filename='Sample_Plot.png'):
        idx =  np.random.randint(0, X.shape[0], n)
        fig = plt.figure(1,figsize=self.figsize)
        for i in range(len(idx)):
            ax = plt.subplot(2,3,i+1)
            if(flag ==  False):
                librosa.display.waveplot(Z[idx[i]],sr=config.audio['SAMPLE_RATE'],alpha=0.6)
                plt.xlabel("time")
                plt.ylabel("amplitude")
            else:
                img = librosa.display.specshow(X[idx[i]],sr=config.audio['SAMPLE_RATE'],y_axis='mel', x_axis='time', hop_length=config.audio['HOP_LENGTH']) #
                fig.colorbar(img, ax=ax, format='%+2.0f dB')
                plt.title(CLASSES[y[idx[i]]])
        plt.tight_layout()
        if(flag ==  False):
            plt.savefig(f'{self.figpath}/{filename}', bbox_inches ="tight", dpi=self.dpi)
        else:
            if(save):
                plt.savefig(f'{self.figpath}/{filename}', bbox_inches ="tight", dpi=self.dpi)
        plt.show()

if __name__ == '__main__':
    """
        This code allows us to preprocess the dataset using varied padding techniques to make the input sequence into fixed length
     """

    pass