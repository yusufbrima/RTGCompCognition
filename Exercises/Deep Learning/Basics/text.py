import numpy as np
import matplotlib.pyplot as plt 
from sklearn.datasets import make_classification,make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import os 

""" We are setting global constants """
NUM_CLASSES =  2
NUM_SAMPLES =  1000
NUM_FEATURES =  20
EPOCHS =  30
BATCH_SIZE =  32

DATA_PATH =  os.path.join(os.getcwd(),"data")

if __name__ == "__main__":
    """  We are making the model workhorse here"""

    #Step 1, define a basic dataset aka fake one 

    df =  pd.read_csv(os.path.join(DATA_PATH,'spam'),sep="\t",names=["Status","Message"])
    df.loc[df["Status"] == "spam", "Status"] = 0
    df.loc[df["Status"] == "ham", "Status"] = 1

    X =  df['Message'].copy()
    y =  df['Status'].copy()
    # print(df["Status"].value_counts())


    # print(X.shape)

    # Next, we want to count vectorize the textual features


    # X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.2,random_state=np.random.seed(1992))

    cv =  CountVectorizer().fit_transform(X)

    X =  cv.toarray()
    y=y.astype('int')
    
    X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.2,random_state=np.random.seed(1992))


    model =  MultinomialNB()



    # model =  RandomForestClassifier(n_estimators=100)

    model.fit(X_train,Y_train)

    # # Making prediction

    Y_hat =  model.predict(X_test)

    print(model.score(X_test,Y_test))

    print(confusion_matrix(Y_test,Y_hat))


