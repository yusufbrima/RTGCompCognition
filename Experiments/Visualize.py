import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 12
FRAME_LENGHT =  1024
SAMPLE_RATE =  44100
HOP_LENGTH =  512
figures = "./Figures/"

def plot_components(X_pca,y,CLASSES, x_str='Component 1', y_str='Component 2', str_title="PCA", title="title.pdf"):
  fig = plt.figure(1,figsize=(10,6))
  ax =  fig.add_subplot(111)
  scatter = ax.scatter(X_pca[:,0],X_pca[:,1], c=list(y))
  ax.set_xlabel(x_str)
  ax.set_ylabel(y_str)
  ax.set_title(str_title)
  # ax.set_zlabel("PCA Component 3")
  ax.legend(handles=scatter.legend_elements()[0], labels=CLASSES,bbox_to_anchor=(1.0, 1.0))
  plt.savefig(f'{figures}/{title}', bbox_inches ="tight", dpi=100)
  plt.show()

def display(X,Z,y,CLASSES,n=6,flag =  False, save=True,filename='Sample_Plot.png'):
  idx =  np.random.randint(0, X.shape[0], n)
  fig = plt.figure(1,figsize=(24,8))
  for i in range(len(idx)):
    ax = plt.subplot(2,3,i+1)
    if(flag ==  False):
      librosa.display.waveplot(Z[idx[i]],sr=SAMPLE_RATE,alpha=0.6)
      plt.xlabel("time")
      plt.ylabel("amplitude")
    else:
      img = librosa.display.specshow(X[idx[i]],sr=SAMPLE_RATE,y_axis='mel', x_axis='time', hop_length=HOP_LENGTH) #
      fig.colorbar(img, ax=ax, format='%+2.0f dB')
    plt.title(CLASSES[y[idx[i]]])
  plt.tight_layout()
  if(flag ==  False):
    plt.savefig(f'{figures}/{filename}', bbox_inches ="tight", dpi=300)
  else:
    if(save):
      plt.savefig(f'{figures}/{filename}', bbox_inches ="tight", dpi=300)
  plt.show()
if __name__ == '__main__':
    """
        This code allows us to preprocess the dataset using varied padding techniques to make the input sequence into fixed length
     """
    pass