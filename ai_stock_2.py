# coding=utf-8
import numpy as np
import pandas as pd
import keras
from keras import regularizers
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
# ai_stock_2:簡易なDeepLearningによる株価予測


def ai_model(input_shape):
    model = Sequential()
    he_normal = keras.initializers.he_normal()

    model.add(Dense(128, input_shape=input_shape, kernel_initializer=he_normal))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, kernel_initializer=he_normal))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1))

    return model


def make_data(y):
    seq_len = 1
    y_data = []
    for i in range(len(y) - seq_len):
        y_data.append(y[i + seq_len])

    return y_data


data_shape = (9,)
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
x = np.array(x[:-1])
y = np.array(make_data(y))
# split the data to train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
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
