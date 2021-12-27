import numpy as np
import matplotlib.pyplot as plt 
from sklearn.datasets import make_classification,make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA,TruncatedSVD
import tensorflow.keras as keras
import tensorflow.keras.backend as K 
import seaborn as sn
import pandas as pd
import os 

""" We are setting global constants """
NUM_CLASSES =  2
NUM_SAMPLES =  1000
NUM_FEATURES =  20
EPOCHS =  30
BATCH_SIZE =  32

MODEL_PATH =  os.path.join(os.getcwd(),"models")


if __name__ == "__main__":
    """  We are making the model workhorse here"""

    #Step 1, define a basic dataset aka fake one 

    X,y =  make_classification(n_samples=NUM_SAMPLES,n_features = NUM_FEATURES,n_classes=NUM_CLASSES,random_state=np.random.seed(1992))


    input_shape =  (NUM_FEATURES,)


    X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.1,random_state=np.random.seed(1992))


    model = RandomForestClassifier(n_estimators=100,verbose=1)


    pca =  PCA(n_components=10,whiten=True)

    pca.fit(X_train).transform(X_train)

    # Step2, load model

