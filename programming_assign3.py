import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from os import path, getcwd, chdir

class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epochs, logs={}):
        if(logs.get('accuracy')>= 0.998):
            print("Reached 99.8'%' accuracy so cancelling training!\n")
            self.model.stop_training = True

def visualize_data(x,y):
    plt.imshow(x)
    plt.show()
    print(y)

def normalize_data(X_train, X_test):
    return X_train/255.0, X_test/255.0

def train_mnist_conv():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    visualize_data(X_train[0].reshape(28, 28), y_train[0])

    X_train, X_test = normalize_data(X_train, X_test)

    X_train, X_test = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1), X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

    model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', input_shape = (28, 28, 1)),
                                        tf.keras.layers.MaxPooling2D(2, 2),
                                        tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
                                        tf.keras.layers.MaxPooling2D(2, 2),
                                        tf.keras.layers.Flatten(),
                                        tf.keras.layers.Dense(1024, activation = tf.nn.relu),
                                        tf.keras.layers.Dense(1024, activation = tf.nn.relu),
                                        tf.keras.layers.Dense(10, activation= tf.nn.softmax) ])

    model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy',metrics=['accuracy'])

    callbacks = MyCallback()
    history = model.fit(X_train, y_train, epochs = 10, callbacks = [callbacks])
    return history.epoch, history.history['accuracy'][-1]
    

if __name__ == "__main__":

    _, _ = train_mnist_conv()
    

   

