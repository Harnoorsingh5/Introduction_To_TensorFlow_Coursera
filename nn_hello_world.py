import tensorflow as tf
import numpy as np
from tensorflow import keras

if  __name__ == "__main__":
    
    #initializing model
    model = tf.keras.Sequential([keras.layers.Dense(units = 1, input_shape=[1])])
    model.compile(optimizer='sgd', loss='mean_squared_error')

    # providing data
    X = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    y = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

    #fitting data to model
    model.fit(X, y, epochs = 500)

    print(model.predict([10.0]))