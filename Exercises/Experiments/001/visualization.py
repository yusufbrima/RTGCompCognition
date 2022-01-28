import numpy as np
import matplotlib.pyplot as plt

def display(array1, array2,sr):
    """
    Displays ten random images from each one of the supplied arrays.
    """

    n = 5

    indices = np.random.randint(len(array1), size=n)
    images1 = array1[indices, :]
    images2 = array2[indices, :]

    plt.figure(figsize=(20, 14))
    for i, (image1, image2) in enumerate(zip(images1, images2)):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(image1.reshape( 1025, 173))
        # librosa.display.specshow(image1.reshape( 1025, 173), hop_length=HOP_LENGTH,sr=sr[i] )
        # plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(image2.reshape( 1025, 173))
        # librosa.display.specshow(image2.reshape( 1025, 173), hop_length=HOP_LENGTH,sr=sr[i] )
        # plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()

if __name__ == '__main__':
    """ This script implements the auto encoder module that removes noise from the input signal"""
    pass 