import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras as keras
from tensorflow.keras.applications import VGG16, ResNet50,VGG19,ResNet50V2,EfficientNetB0
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import utils as np_utils


def make_model(input_shape, output_nums):
    base_model = VGG16(weights=None, include_top=False, input_shape=input_shape)
    x = base_model.output
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(output_nums, activation='softmax')(x)
    model = keras.Model(inputs=base_model.input, outputs=x)
    optim_params = dict(learning_rate = 0.003,momentum = 0.9394867962846013,decay = 0.0003)
    model.compile(loss='sparse_categorical_crossentropy',optimizer=keras.optimizers.SGD(**optim_params),metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy"),keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='sparse_top_5_categorical_accuracy')])
    return model

def build_model(input_shape,num_output):
    
    inputs = keras.Input(shape=input_shape)
    x =  keras.layers.Dense(32,activation="relu",kernel_regularizer=keras.regularizers.l2(0.2))(inputs)

    x =   keras.layers.Dense(64,activation="relu",kernel_regularizer=keras.regularizers.l2(0.2))(x)
    x =  keras.layers.Dropout(0.3)(x)

    x =   keras.layers.Dense(128,activation="relu",kernel_regularizer=keras.regularizers.l2(0.2))(x)
    x =  keras.layers.Dropout(0.3)(x)

    x =   keras.layers.Dense(256,activation="relu",kernel_regularizer=keras.regularizers.l2(0.2))(x)
    x =  keras.layers.Dropout(0.3)(x)

    x =   keras.layers.Dense(128,activation="relu",kernel_regularizer=keras.regularizers.l2(0.2))(x)
    x =  keras.layers.Dropout(0.3)(x)

    x =   keras.layers.Dense(64,activation="relu",kernel_regularizer=keras.regularizers.l2(0.2))(x)
    x =  keras.layers.Dropout(0.3)(x)

    x =   keras.layers.Dense(32,activation="relu",kernel_regularizer=keras.regularizers.l2(0.2))(x)
    x =  keras.layers.Dropout(0.3)(x)

    outputs =  keras.layers.Dense(num_output,activation="softmax")(x)

    model =  keras.Model(inputs=inputs,outputs=outputs,name="deepspeaker")

    model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])
    return model

if __name__ == '__main__':
    """
        This code allows us to preprocess the dataset using varied padding techniques to make the input sequence into fixed length
     """
    pass 