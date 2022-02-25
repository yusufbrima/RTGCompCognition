from matplotlib.pyplot import figure, show
import numpy as np
from numpy.random import rand
from sklearn.decomposition import  PCA,TruncatedSVD,FastICA
import sounddevice as sd
import librosa
import config
fs = config.audio['SAMPLE_RATE']


f =  config.data['file_path']
data =  np.load(f, allow_pickle=True)
X =  data['x']
y = data['y']
CLASSES = list(data['c'])

embedding = PCA(n_components=2)
X_scaled =  (X- X.mean(axis=0, keepdims=True))/X.std(axis=0,keepdims=True)
X_transformed = embedding.fit_transform(X_scaled.reshape(X_scaled.shape[0],-1))
# X_transformed.shape
if 1: # picking on a scatter plot (matplotlib.collections.RegularPolyCollection)
    def tracker(event):
        ind = event.ind
        
        ax3.clear()
        ax3.cla()
        idx = np.random.randint(0, X_scaled.shape[0])
        col = ax3.imshow(X_scaled[idx])
        ax3.set_axis_off()
        
        y_inv = librosa.griffinlim(X[idx],hop_length=256)
        sd.play(y_inv, fs,blocking=True)

        
        ax2.clear()
        ax2.cla()
        col = ax2.plot(y_inv)
        ax2.set_axis_off() 
        fig.canvas.draw()
        print('onpick3 scatter:', ind, np.take(X_transformed[:,0], ind), np.take(X_transformed[:,1], ind))

    fig = figure(figsize=(12,6))
    ax1 = fig.add_subplot(2,1,1)
    col = ax1.scatter(X_transformed[:,0], X_transformed[:,1],c=list(y), picker=True)
    ax3 = fig.add_subplot(2, 2, 4)
    ax2 = fig.add_subplot(2, 2, 3)
    #fig.savefig('pscoll.eps')
    fig.canvas.mpl_connect('pick_event', tracker)
    

show()
# plt.gcf().canvas.draw_idle()