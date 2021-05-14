from tensorflow import kerasfrom
from tensorflow.keras.layers import Dense


model = keras.Sequential()
model.add(*Dense(units=1, input_shape=)1,)))
model.compile(optimizer='sgd', loss='mse')
model.summary()
