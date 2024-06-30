import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, r2_score

from keras.models import Sequential
from keras.layers import LSTM, Dense, Reshape

df = pd.read_csv('/data.csv', index_col='Date and Time')

data = df.filter(['annualised consumption GUESS, TWh'])

data.index = pd.to_datetime(data.index)

dataset = data.values

training_data_len = math.ceil(len(dataset)*.9)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

scaled_data

train_data = scaled_data[:training_data_len, :]

x_train = []
y_train = []

for i in range (60, len(train_data)):
  x_train.append(train_data[i-60:i, 0])
  y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

"""Create Test Dataset"""

test_data = scaled_data[training_data_len-60:, :]
x_test=[]
y_test=dataset[training_data_len:, :]
y_test_scaled=scaled_data[training_data_len:, :]

for i in range (60, len(test_data)):
  x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

import tensorflow as tf

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

"""Model"""

model = Sequential()

model.add(LSTM(64,activation='tanh', return_sequences=True, input_shape = (x_train.shape[1], 1)))
model.add(LSTM(64,activation='tanh'))
model.add(Dense(32,activation='tanh'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

model.summary()

history = model.fit(x_train, y_train, epochs=30, batch_size=16, validation_split=0.1, verbose=1, callbacks = [callback])

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Val loss')
plt.legend()
plt.show()

predictions = model.predict(x_test)

predictions = scaler.inverse_transform(predictions)

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, predictions)
mse

rmse = np.sqrt(mean_squared_error(y_test, predictions))
rmse

r2 = r2_score(y_test, predictions)
r2

#Plot the data
train=data[: training_data_len]
valid=data[training_data_len:]
valid['Predictions'] = predictions
#Visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('annualised consumption GUESS, TWh', fontsize=18)
plt.plot(valid [['annualised consumption GUESS, TWh', 'Predictions']])
plt.legend(['Val', 'Predictions'], loc='lower right')
plt.show()
print("Root-Mean-Square Error: {}".format(rmse))
print("R-Squared Value: {}".format(r2))

