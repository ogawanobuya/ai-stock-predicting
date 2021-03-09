# coding=utf-8
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
# ai_stock_1:重回帰分析による株価予測


def make_data(y):
    seq_len = 1
    y_data = []
    for i in range(len(y) - seq_len):
        y_data.append(y[i + seq_len])

    return y_data


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
# make model
model = LinearRegression()
model.fit(x_train, y_train)
# make prediction
y_pred = model.predict(x_test)
print("Test score:", r2_score(y_test, y_pred))
# show prediction
y_pred = yScaler.inverse_transform(y_pred)
y_test = yScaler.inverse_transform(y_test)
print ("Pred\n", y_pred[:3])
print ("Answ\n", y_test[:3])
