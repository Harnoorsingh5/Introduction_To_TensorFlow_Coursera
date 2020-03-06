import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

def visualize_data(x,y):
    plt.imshow(x)
    plt.show()
    print(y)

def normalize_data(X_train, X_test):
    return X_train/255.0, X_test/255.0

if __name__ == "__main__":

    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    visualize_data(X_train[0].reshape(28, 28), y_train[0])

    X_train, X_test = normalize_data(X_train, X_test)

    model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape = (28,28)),
                                        tf.keras.layers.Dense(128, activation = tf.nn.relu),
                                        tf.keras.layers.Dense(10, activation= tf.nn.softmax) ])

    model.compile(optimizer='sgd', loss = 'sparse_categorical_crossentropy')

    model.fit(X_train, y_train, epochs = 5)

    model.evaluate(X_test, y_test)

