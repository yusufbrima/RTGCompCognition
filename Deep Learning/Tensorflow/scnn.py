import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.imagenet_utils import decode_predictions 
from tensorflow.keras.applications import VGG16, ResNet50,ResNet50V2,ResNet152V2, VGG19, DenseNet121, InceptionV3, Xception,EfficientNetB7,DenseNet169,DenseNet201,InceptionResNetV2,EfficientNetB7
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import utils as np_utils
warnings.simplefilter(action='ignore', category=FutureWarning)

DATA_PATH = "/net/projects/scratch/winter/valid_until_31_July_2022/ybrima/data/data_genre.npz"
# DATA_PATH = "/net/projects/scratch/winter/valid_until_31_July_2022/ybrima/data/data_chimp.npz"
REPORT_PATH = "/net/store/cv/users/ybrima/RTGCompCog/Deep Learning/Tensorflow/reports/report.csv"
NUM_CLASSES =  10

def load_data(data_path):
    """Loads training dataset from json file.

        :param data_path (str): Path to compressed npz file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """
    
    # load data
    data =  np.load(data_path)

    X,y = data['x'],data['y']
    return X, y

def train_test():
        # get train, validation, test splits
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.10, 0.2)

    # create network
    input_shape = (X_train.shape[1], X_train.shape[2], 1)

    models = [ VGG16, ResNet50,ResNet50V2,ResNet152V2, VGG19, DenseNet121,EfficientNetB7,DenseNet169,DenseNet201,InceptionResNetV2,EfficientNetB7]
    m_names = [ "VGG16", "ResNet50","ResNet50V2","ResNet152V2", "VGG19", "DenseNet121","EfficientNetB7","DenseNet169","DenseNet201","InceptionResNetV2","EfficientNetB7"]
    res = {"model":[], "val_accuracy":[], "test_accuracy":[] }
    for i in range(len(models)):
        K.clear_session()
        # EfficientNetB7,DenseNet169,DenseNet201,InceptionResNetV2,EfficientNetB7
        base_model = models[i](weights=None, include_top=False, input_shape=input_shape)

        x = base_model.output
        x = Flatten()(x)
        x = Dense(NUM_CLASSES, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=x)

        optim_params = dict(learning_rate = 0.0001,momentum = 0.9394867962846013,decay = 0.0001)


        model.compile(loss='sparse_categorical_crossentropy',optimizer=keras.optimizers.SGD(**optim_params),metrics=['accuracy'])
            # # train model
        history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=30)

        # # plot accuracy/error for training and validation
        # plot_history(history)

        # evaluate model on test set
        _, test_acc = model.evaluate(X_test, y_test, verbose=2)
        res['model'].append(m_names[i])
        res['test_accuracy'].append(test_acc)
        #res['val_accuracy'].append(test_acc)
        print('\nTest accuracy:', test_acc)

        # pick a sample to predict from the test set
        X_to_predict = X_test[100]
        y_to_predict = y_test[100]
        print(f"Val_accuracy {np.array(history.history['val_accuracy']).mean()}")
        # predict sample
        predict(model, X_to_predict, y_to_predict)
        break
    df = pd.DataFrame(res)
    df.to_csv(REPORT_PATH,df,index=False)
    print(f"Experiment completed successfully, report saved to {REPORT_PATH}")
def normalize(X):
    return (X -  np.mean(X, axis=0))/np.std(X, axis= 0)


def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs

        :param history: Training history of model
        :return:
    """

    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="val accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="val error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()


def prepare_datasets(test_size, validation_size):
    """Loads data and splits it into train, validation and test sets.

    :param test_size (float): Value in [0, 1] indicating percentage of data set to allocate to test split
    :param validation_size (float): Value in [0, 1] indicating percentage of train set to allocate to validation split

    :return X_train (ndarray): Input training set
    :return X_validation (ndarray): Input validation set
    :return X_test (ndarray): Input test set
    :return y_train (ndarray): Target training set
    :return y_validation (ndarray): Target validation set
    :return y_test (ndarray): Target test set
    """

    # load data
    X, y = load_data(DATA_PATH)
    X = normalize(X)
    # create train, validation and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,shuffle=True)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size,shuffle=True)

    # add an axis to input sets
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test



def predict(model, X, y):
    """Predict a single sample using the trained model

    :param model: Trained classifier
    :param X: Input data
    :param y (int): Target
    """

    # add a dimension to input data for sample - model.predict() expects a 4d array in this case
    X = X[np.newaxis, ...] # array shape (1, 130, 32, 1)

    # perform prediction
    prediction = model.predict(X)

    # get index with max value
    predicted_index = np.argmax(prediction, axis=1)

    print("Target: {}, Predicted label: {}".format(y, predicted_index))


if __name__ == "__main__":
    train_test()