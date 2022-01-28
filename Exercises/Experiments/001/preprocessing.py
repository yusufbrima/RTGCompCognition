import numpy as np


def noise(array):
    """
    Adds random noise to each image in the supplied array.
    """

    noise_factor = 0.4
    noisy_array = array + noise_factor * np.random.normal(
        loc=0.0, scale=1.0, size=array.shape
    )

    return np.clip(noisy_array, 0.0, 1.0)

def preprocess(array):
    """
    Normalizes the supplied array and reshapes it into the appropriate format.
    """

    array = array.astype("float32") 
    array = np.reshape(array, (len(array), 1025, 173, 1))
    return array

if __name__ == '__main__':
    """ This script implements the auto encoder module that removes noise from the input signal"""
    pass 