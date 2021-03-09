# coding=utf-8
import numpy as np
import pandas as pd
import keras
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM, TimeDistributed
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
# ai_stock_3:RNNによる株価予測


def ai_model(input_shape):
    model = Sequential()
    he_normal = keras.initializers.he_normal()
    # build layer
    model.add(LSTM(64, dropout=0.3, batch_input_shape=(None, input_shape[0], input_shape[1]), return_sequences=True))
    model.add(TimeDistributed(Dense(128, kernel_initializer=he_normal)))
    model.add(LSTM(64, dropout=0.3, recurrent_dropout=0.3, return_sequences=True))
    model.add(TimeDistributed(Dense(128, kernel_initializer=he_normal)))
    model.add(LSTM(64, dropout=0.3, recurrent_dropout=0.3, return_sequences=False))
    model.add(Activation('relu'))
    model.add(Dense(1))

    return model


def make_data(data_shape, x, y):
    seq_len = data_shape[0]
    x_data, y_data = [], []
    for i in range(len(x) - seq_len):
        x_data.append(x[i:i + seq_len])
        y_data.append(y[i + seq_len])

    return x_data, y_data


data_shape = (16, 9)
# make model
model = ai_model(data_shape)
# get Data
csv_data = pd.read_csv('data/test_data.csv')
x = csv_data.drop(['Date'], axis=1).values
y = csv_data['^GSPC'].values.reshape(-1, 1)
# Normalize the numerical values
xScaler = StandardScaler()
yScaler = StandardScaler()
x = xScaler.fit_transform(x)
y = yScaler.fit_transform(y)
# split the data to train and test
x_data, y_data = make_data(data_shape, x, y)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
# train the model
model.compile(loss='mse',
              optimizer=Adam())
model.fit(x_train, y_train,
          batch_size=16, epochs=5, verbose=1)
# make prediction
y_pred = model.predict(x_test)
print("Test score:", r2_score(y_test, y_pred))
# show prediction
y_pred = yScaler.inverse_transform(y_pred)
y_test = yScaler.inverse_transform(y_test)
print ("Pred\n", y_pred[:3])
print ("Answ\n", y_test[:3])
