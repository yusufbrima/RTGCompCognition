import tensorflow.keras as keras


def ae(input_shape):

  inputs = keras.Input(shape=input_shape)
  # Encoder
  x = keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
  x = keras.layers.MaxPooling2D((2, 2), padding="same")(x)
  x = keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
  x = keras.layers.MaxPooling2D((2, 2), padding="same")(x)

  # Decoder 
  x = keras.layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
  x = keras.layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
  x = keras.layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)
  
  model = keras.Model(inputs = inputs, outputs = x, name="autoencoder")
  model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
  return model

if __name__ == '__main__':
    """ This script implements the auto encoder module that removes noise from the input signal"""
    pass 