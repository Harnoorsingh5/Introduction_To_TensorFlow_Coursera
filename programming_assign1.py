import tensorflow as tf
import numpy as np
from tensorflow import keras

if  __name__ == "__main__":
    
    model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer='sgd', loss='mean_squared_error')

    # providing data
    xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0], dtype=float)
    ys = np.array([100000.0, 150000.0, 200000.0, 250000.0, 300000.0, 350000.0, 550000.0], dtype=float)/1000.0

    #fitting data to model
    model.fit(xs, ys, epochs=500)
    print(model.predict([7.0]))
