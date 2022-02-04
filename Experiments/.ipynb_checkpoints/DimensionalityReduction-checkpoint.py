import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import  PCA,TruncatedSVD,FastICA

def reduce_dimensions(model,X, n_components=3):
    """ This functions using the selected dimensionality reduction technique to give a low dimensional representation of the data """
    
    model = model(n_components=3)
    
    return  model.fit_transform(X.reshape(X.shape[0],-1))
if __name__ == '__main__':
    """
        This code allows us to preprocess the dataset using varied padding techniques to make the input sequence into fixed length
     """
    pass