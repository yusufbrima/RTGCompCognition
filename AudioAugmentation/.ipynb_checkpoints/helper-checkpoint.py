import librosa.display
import matplotlib.pyplot as plt


def plot_signal_and_augmented_singal(signal,asignal,sr):
    fig,ax =  plt.subplots(ncols=2, nrows=1)
    librosa.display.waveplot(signal, sr=sr, ax=ax[0])
    ax[0].set(title="Original Signal")
    
    librosa.display.waveplot(asignal, sr=sr, ax=ax[1])
    ax[1].set(title="Augmented Signal")
    plt.tight_layout()
    plt.show()
    