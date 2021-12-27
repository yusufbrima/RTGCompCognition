import numpy as np
import matplotlib.pyplot as plt 
from sklearn.datasets import make_classification,make_regression
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import tensorflow.keras.backend as K 
from sklearn.metrics import confusion_matrix,classification_report
import seaborn as sn
import pandas as pd
import os 

""" We are setting global constants """
NUM_CLASSES =  2
NUM_SAMPLES =  10000
NUM_FEATURES =  20
EPOCHS =  30
BATCH_SIZE =  32

MODEL_PATH =  os.path.join(os.getcwd(),"models")


def make_model(input_shape,output):
    K.clear_session()
    model =  keras.Sequential([
        keras.layers.Dense(32, activation="relu",input_shape=input_shape,kernel_regularizer=keras.regularizers.l2(0.02)),
        keras.layers.Dropout(0.3),
        keras.layers.BatchNormalization(),

        keras.layers.Dense(64, activation="relu",kernel_regularizer=keras.regularizers.l2(0.02)),
        keras.layers.Dropout(0.3),
        keras.layers.BatchNormalization(),

        keras.layers.Dense(128, activation="relu",kernel_regularizer=keras.regularizers.l2(0.02)),
        keras.layers.Dropout(0.3),
        keras.layers.BatchNormalization(),


        keras.layers.Dense(64, activation="relu",kernel_regularizer=keras.regularizers.l2(0.02)),
        keras.layers.Dropout(0.3),
        keras.layers.BatchNormalization(),

        keras.layers.Dense(32, activation="relu",kernel_regularizer=keras.regularizers.l2(0.02)),
        keras.layers.Dropout(0.3),
        keras.layers.BatchNormalization(),

        keras.layers.Flatten(),
        keras.layers.Dense(output,activation="softmax")
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),loss="sparse_categorical_crossentropy",metrics=["accuracy"])

    print("Model created successfully")
    print(model.summary())
    return model

def plot_features(X,y):
    scatter = plt.scatter(X[:,np.random.randint(0,X.shape[1])],X[:,np.random.randint(0,X.shape[1])], c=list(y))
    plt.legend(handles=scatter.legend_elements()[0], labels=list(np.unique(y)))
    plt.show()

def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs

        :param history: Training history of model
        :return:
    """

    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()

if __name__ == "__main__":
    """  We are making the model workhorse here"""

    #Step 1, define a basic dataset aka fake one 

    X,y =  make_classification(n_samples=NUM_SAMPLES,n_features = NUM_FEATURES,n_classes=NUM_CLASSES,random_state=np.random.seed(1992))

    input_shape =  (NUM_FEATURES,)

    # plot_features(X,y)

    # Step 2: make a model

    model = make_model(input_shape=input_shape, output=NUM_CLASSES)


    # Step 4: split the dataset 

    # X_train,X_valid,Y_train,Y_valid = train_test_split(X,y,test_size=0.2,random_state=np.random.seed(1992))

    X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.1,random_state=np.random.seed(1992))


    # Step 5: Train the model

    # history = model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), batch_size=BATCH_SIZE, epochs=EPOCHS)
    history = model.fit(X_train, Y_train, validation_split=0.2, batch_size=BATCH_SIZE, epochs=EPOCHS)
    # # plot accuracy/error for training and validation
    
    plot_history(history)

    # Step 6: Use the model to make predictions

    predictions =  model.predict(X_test,batch_size=BATCH_SIZE,verbose=1)

    pred_classes =  np.argmax(predictions,axis=1)

    con_mat =  confusion_matrix(Y_test,pred_classes)

    class_labels = ["Cats","Dogs"]
    df_cm = pd.DataFrame(con_mat, class_labels, class_labels)
    plt.figure(figsize=(10,7))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16},fmt="g",cmap='Blues') # font size
    plt.xlabel('\nPredicted Values')
    plt.ylabel('Actual Values')
    plt.show()

    model.save(os.path.join(MODEL_PATH,"model.h5"))



    