import numpy as np
import matplotlib.pyplot as plt 
from sklearn.datasets import make_classification,make_regression
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import tensorflow.keras.backend as K 
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix,classification_report
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

    # Step2, load model

    model =  keras.models.load_model(os.path.join(MODEL_PATH,"model.h5"))


    predictions =  model.predict(X,batch_size=BATCH_SIZE,verbose=1)

    pred_classes =  np.argmax(predictions,axis=1)

    con_mat =  confusion_matrix(y,pred_classes)

    class_labels = ["Cats","Dogs"]
    df_cm = pd.DataFrame(con_mat, class_labels, class_labels)
    plt.figure(figsize=(10,7))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16},fmt="g",cmap='Blues') # font size
    plt.xlabel('\nPredicted Values')
    plt.ylabel('Actual Values')
    plt.show()

