import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

def QModel(input_dim, output_dim, lr):
    model = Sequential()
    model.add(Dense(8, input_dim=input_dim, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(output_dim))
    model.compile(loss="mse", optimizer=Adam(lr))

    return model
